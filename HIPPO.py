seed = 123
import numpy as np
np.random.seed(seed)
import random as rn
rn.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)
import torch

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from utils.config import args
import random
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
import math
cudnn.benchmark = True

from utils.bar_show import progress_bar
from src.cmdataset import CMDataset
import scipy
import scipy.spatial

from sklearn.cluster import AgglomerativeClustering

import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

best_acc = 0  # best test accuracy
start_epoch = 0

args.log_dir = os.path.join(args.root_dir, 'logs', args.log_name)
args.ckpt_dir = os.path.join(args.root_dir, 'ckpt', args.pretrain_dir)

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)

class Net(nn.Module):
    def __init__(self, input_dim, hash_dim, hidden_dim=8192, num_hidden_layers=3, encoder=[2,1], norm=True):
        super(Net, self).__init__()
        self.norm = norm
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=encoder[0], batch_first=True),
            num_layers=encoder[1]
        )

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_hidden_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, hash_dim))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.transformer(x.unsqueeze(0)).squeeze(0)
        out = self.fc(x)
        out = torch.tanh(out)
        if self.norm:
            out = F.normalize(out, p=2, dim=1)

        return out

def hierarchical_clustering(features, num_clusters):
    clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    clustering.fit(features)
    labels = clustering.labels_

    centers = np.zeros((num_clusters, features.shape[1]))
    for cluster_idx in range(num_clusters):
        cluster_points = features[labels == cluster_idx]
        centers[cluster_idx] = np.mean(cluster_points, axis=0)

    return centers

def soft_labels_and_binary_pseudo_labels(features, centers, threshold):
    dists = torch.cdist(torch.tensor(features, dtype=torch.float32).cuda(),
                        torch.tensor(centers, dtype=torch.float32).cuda(), p=2)
    soft_assign = F.softmax(-dists, dim=1)
    avg_dists = torch.mean(dists, dim=1, keepdim=True)
    avg_dists = avg_dists * threshold
    binary_labels = (dists < avg_dists).float()

    return soft_assign, binary_labels

class IncorrectSampleMemory:
    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.memory = []

    def size(self):
        return len(self.memory)

    def add(self, image, text, image_center, text_center, image_raw, text_raw):
        image = image.detach()
        text = text.detach()
        image_center = image_center.detach()
        text_center = text_center.detach()
        image_raw = image_raw.detach()
        text_raw = text_raw.detach()
        for i in range(image.size(0)):
            if len(self.memory) >= self.max_size:
                self.memory.pop(0)
            self.memory.append((
                image[i],
                text[i],
                image_center[i],
                text_center[i],
                image_raw[i],
                text_raw[i]
            ))

    def sample(self, batch_size):
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        image, text, image_center, text_center, image_raw, text_raw = zip(*batch)
        return torch.stack(image), torch.stack(text), torch.stack(image_center), torch.stack(text_center), torch.stack(
            image_raw), torch.stack(text_raw)

class CrossModalTripletLoss_error(nn.Module):
    def __init__(self, margin=1.0):
        super(CrossModalTripletLoss_error, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, image_hash, text_hash, labels):
        batch_size = image_hash.size(0)
        iszero = False
        dist_image_text = torch.cdist(image_hash, text_hash, p=2)
        dist_text_image = torch.cdist(text_hash, image_hash, p=2)

        positive_mask = torch.eye(batch_size, dtype=torch.bool, device=image_hash.device)
        label_intersection = torch.mm(labels.float(), labels.float().T)
        negative_mask = label_intersection == 0
        num_negatives_per_sample = negative_mask.sum(dim=1)

        if num_negatives_per_sample.min().item() == 0:
            random_negative_mask = ~positive_mask
            negative_mask = random_negative_mask
            iszero = True
            
        num_negatives_per_sample = negative_mask.sum(dim=1)

        if iszero:
            error_num = 2
        else:
            error_num = min(num_negatives_per_sample.min().item(), 4)

        sampled_negative_dist_image_to_text = []
        sampled_negative_dist_text_to_image = []

        for i in range(batch_size):
            negative_indices = negative_mask[i].nonzero(as_tuple=True)[0]
            sampled_indices = torch.randperm(negative_indices.size(0))[:error_num]
            sampled_negative_dist_image_to_text.append(dist_image_text[i, negative_indices[sampled_indices]])

            negative_indices = negative_mask[:, i].nonzero(as_tuple=True)[0]
            sampled_indices = torch.randperm(negative_indices.size(0))[:error_num]
            sampled_negative_dist_text_to_image.append(dist_text_image[i, negative_indices[sampled_indices]])

        sampled_negative_dist_image_to_text = torch.stack(
            sampled_negative_dist_image_to_text)
        sampled_negative_dist_text_to_image = torch.stack(
            sampled_negative_dist_text_to_image)

        positive_dist_image_to_text = dist_image_text[positive_mask]
        positive_dist_text_to_image = dist_text_image[positive_mask]

        loss_image_to_text = self.relu(
            positive_dist_image_to_text.unsqueeze(1) - sampled_negative_dist_image_to_text + self.margin
        ).mean()
        loss_text_to_image = self.relu(
            positive_dist_text_to_image.unsqueeze(1) - sampled_negative_dist_text_to_image + self.margin
        ).mean()

        triplet_loss = loss_image_to_text + loss_text_to_image
        return triplet_loss
    
class AuxiliaryNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=2):
        super(AuxiliaryNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

def attraction_loss(hash_code1, hash_code2, margin=0.5):
    distance = torch.sqrt(((hash_code1 - hash_code2) ** 2).sum(dim=1) + 1e-8)
    loss = torch.clamp(distance - margin, min=0).mean()
    return loss


centers = {
    "image": None,
    "text": None
}


def main():
    print('===> Preparing data ..')
    train_dataset = CMDataset(
        args.data_name,
        return_index=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    retrieval_dataset = CMDataset(
        args.data_name,
        partition='retrieval'
    )
    retrieval_loader = torch.utils.data.DataLoader(
        retrieval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_dataset = CMDataset(
        args.data_name,
        partition='test'
    )
    query_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_model = Net(
        input_dim=train_dataset.imgs.shape[1],
        hash_dim=args.bit,
        hidden_dim=8192,
        num_hidden_layers=args.num_hiden_layers[0],
        encoder=[args.num_hiden_layers[2],args.num_hiden_layers[3]],
        norm=True
    ).cuda()
    text_model = Net(
        input_dim=train_dataset.text_dim,
        hash_dim=args.bit,
        hidden_dim=8192,
        num_hidden_layers=args.num_hiden_layers[1],
        encoder=[args.num_hiden_layers[4],args.num_hiden_layers[5]],
        norm=True
    ).cuda()
    aux_image_net = AuxiliaryNetwork(args.bit, int(train_dataset.imgs.shape[1]*args.dim1), train_dataset.imgs.shape[1]).cuda()
    aux_text_net = AuxiliaryNetwork(args.bit, int(train_dataset.imgs.shape[1]*args.dim2), train_dataset.text_dim).cuda()

    parameters = list(image_model.parameters()) + list(text_model.parameters()) + list(aux_image_net.parameters()) + list(aux_text_net.parameters())
    wd = args.wd
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=wd)
    lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [300, 600, 900, 1200], gamma=0.1)

    summary_writer = SummaryWriter(args.log_dir)

    if args.resume:
        ckpt = torch.load(os.path.join(args.ckpt_dir, args.resume))
        image_model.load_state_dict(ckpt['image_model_state_dict'])
        text_model.load_state_dict(ckpt['text_model_state_dict'])
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print('===> Load last checkpoint data')
    else:
        start_epoch = 0
        print('===> Start from scratch')

    def set_train():
        image_model.train()
        text_model.train()

    def set_eval():
        image_model.eval()
        text_model.eval()

    tripletLoss = CrossModalTripletLoss_error(margin=args.margin)
    k = args.enk
    en_memory = IncorrectSampleMemory(max_size=k * len(train_loader))
    
    if args.data_name == 'nus':
        categories = 10
    elif args.data_name == 'flickr':
        categories = 24
    elif args.data_name == 'coco':
        categories = 80
    elif args.data_name == 'iapr':
        categories = 255
    
    def train(epoch):
        print('\nEpoch: %d / %d' % (epoch, args.max_epochs))
        set_train()
        train_loss, correct, total = 0., 0., 0.
        for batch_idx, (idx, images, texts, _) in enumerate(train_loader):
            images, texts, idx = [img.cuda() for img in images], [txt.cuda() for txt in texts], [idx.cuda()]
            images_np = np.vstack([img.cpu().numpy() for img in images])
            texts_np = np.vstack([txt.cpu().numpy() for txt in texts])

            new_image_centers = hierarchical_clustering(images_np, categories)
            new_text_centers = hierarchical_clustering(texts_np, categories)
            if centers['image'] is None:
                centers['image'] = new_image_centers
            else:
                temp_image_centers = np.vstack((new_image_centers, centers['image']))
                centers['image'] = hierarchical_clustering(temp_image_centers, categories)

            if centers['text'] is None:
                centers['text'] = new_text_centers
            else:
                temp_text_centers = np.vstack((new_text_centers, centers['text']))
                centers['text'] = hierarchical_clustering(temp_text_centers, categories)

            image_centers = centers['image']
            text_centers = centers['text']

            image_soft_labels, image_binary_labels = soft_labels_and_binary_pseudo_labels(images_np, image_centers, args.threshold)
            text_soft_labels, text_binary_labels = soft_labels_and_binary_pseudo_labels(texts_np, text_centers, args.threshold)
            fused_labels = image_binary_labels.bool() & text_binary_labels.bool()
            fused_labels = fused_labels.long().cuda()

            images_outputs = [image_model(im) for im in images]
            texts_outputs = [text_model(txt.float()) for txt in texts]

            labels_similarity_matrix1 = torch.matmul(fused_labels.float(), fused_labels.float().T)
            labels_similarity_matrix1 = F.normalize(labels_similarity_matrix1, dim=1)
            hash_similarity_matrix = torch.matmul(images_outputs[0], texts_outputs[0].T)
            similarity_difference = torch.abs(labels_similarity_matrix1 - hash_similarity_matrix)

            topk_values, topk_indices_flat = torch.topk(similarity_difference.flatten(), k=k)
            rows, cols = similarity_difference.size()
            row_indices = topk_indices_flat // cols  # Row Index
            col_indices = topk_indices_flat % cols  # Column Index

            topk_indices_nd = (row_indices, col_indices)

            incorrect_samples_idx = list(zip(topk_indices_nd[0].tolist(), topk_indices_nd[1].tolist()))
            incorrect_samples_idx = torch.tensor(incorrect_samples_idx)

            img_indices = incorrect_samples_idx[:, 0]
            text_indices = incorrect_samples_idx[:, 1]

            images_outputs_selected = images_outputs[0][img_indices]
            texts_outputs_selected = texts_outputs[0][text_indices]
            image_incorrect_label = fused_labels[img_indices]
            text_incorrect_label = fused_labels[text_indices]
            images_raw_selected = images[0][img_indices]
            texts_raw_selected = texts[0][text_indices]

            en_memory.add(images_outputs_selected, texts_outputs_selected, image_incorrect_label,
                          text_incorrect_label, images_raw_selected, texts_raw_selected)
            if en_memory.size() < max(args.en1, args.en2):
                image_len = en_memory.size()
                text_len = en_memory.size()
            else:
                image_len = args.en1
                text_len = args.en2
            incorrect_image_outputs, incorrect_text_outputs, incorrect_image_label, incorrect_text_label, incorrect_image_raw, incorrect_text_raw = en_memory.sample(max(args.en1, args.en2))

            incorrect_image_outputs = image_model(incorrect_image_raw[:image_len])
            incorrect_text_outputs = text_model(incorrect_text_raw[:text_len].float())
            
            optimized_image_features = aux_image_net(incorrect_text_outputs)
            optimized_text_features = aux_text_net(incorrect_image_outputs)

            image_target_centers = torch.matmul(incorrect_image_label[:text_len].float(),
                                                torch.tensor(image_centers, dtype=torch.float32).cuda())
            text_target_centers = torch.matmul(incorrect_text_label[:image_len].float(),
                                               torch.tensor(text_centers, dtype=torch.float32).cuda())

            loss_mcm = F.mse_loss(optimized_image_features, image_target_centers) + F.mse_loss(optimized_text_features, text_target_centers)
            loss_att = attraction_loss(images_outputs[0], texts_outputs[0], margin=0.4)
            loss_dis = tripletLoss(images_outputs[0], texts_outputs[0], fused_labels)
            loss = loss_dis * 0.6 + loss_mcm * 0.2 + loss_att * 0.2

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(parameters, 1.)
            optimizer.step()
            train_loss += loss.item()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | LR: %g'
                         % (train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))

            if batch_idx % args.log_interval == 0:
                summary_writer.add_scalar('Loss/train', train_loss / (batch_idx + 1),
                                          epoch * len(train_loader) + batch_idx)
                summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'],
                                          epoch * len(train_loader) + batch_idx)

        print(
            f"loss: {loss.item()}")

    def eval(data_loader):
        imgs, txts, labs = [], [], []
        with torch.no_grad():
            for batch_idx, (images, texts, targets) in enumerate(data_loader):
                images, texts, targets = [img.cuda() for img in images], [txt.cuda() for txt in texts], targets.cuda()

                images_outputs = [image_model(im) for im in images]
                texts_outputs = [text_model(txt.float()) for txt in texts]

                imgs += images_outputs
                txts += texts_outputs
                labs.append(targets)

            imgs = torch.cat(imgs).sign_().cpu().numpy()
            txts = torch.cat(txts).sign_().cpu().numpy()
            labs = torch.cat(labs).cpu().numpy()
        return imgs, txts, labs

    def test(epoch, is_eval=True):
        global best_acc
        set_eval()
        # switch to evaluate mode
        (retrieval_imgs, retrieval_txts, retrieval_labs) = eval(retrieval_loader)
        if is_eval:
            query_imgs, query_txts, query_labs = retrieval_imgs[0: 2000], retrieval_txts[0: 2000], retrieval_labs[0: 2000]
            retrieval_imgs, retrieval_txts, retrieval_labs = retrieval_imgs[0: 2000], retrieval_txts[0: 2000], retrieval_labs[0: 2000]
        else:
            (query_imgs, query_txts, query_labs) = eval(query_loader)

        i2t = fx_calc_map_multilabel_k(retrieval_txts, retrieval_labs, query_imgs, query_labs, k=0, metric='hamming')
        t2i = fx_calc_map_multilabel_k(retrieval_imgs, retrieval_labs, query_txts, query_labs, k=0, metric='hamming')

        avg = (i2t + t2i) / 2.
        print('%s\nImg2Txt: %g \t Txt2Img: %g \t Avg: %g' % ('Evaluation' if is_eval else 'Test',i2t, t2i, (i2t + t2i) / 2.))
        if avg > best_acc:
            print('Saving..')
            state = {
                'image_model_state_dict': image_model.state_dict(),
                'text_model_state_dict': text_model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'Avg': avg,
                'Img2Txt': i2t,
                'Txt2Img': t2i,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(args.ckpt_dir, '%s_%d_best_%s_checkpoint.t7' % (args.arch, args.bit, args.data_name)))
            best_acc = avg
        return i2t, t2i

    def adjust_learning_rate(optimizer, epoch, warmup_epochs=5, base_lr=1e-5, target_lr=1e-4):
        if epoch < warmup_epochs:
            lr = base_lr + (target_lr - base_lr) * epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = target_lr
    
    lr_schedu.step(start_epoch)
    for epoch in range(start_epoch, args.max_epochs):
        adjust_learning_rate(optimizer, epoch, args.warmup_epoch, base_lr=1e-5, target_lr=args.lr)
        train(epoch)
        lr_schedu.step(epoch)
        i2t, t2i = test(epoch)
        avg = (i2t + t2i) / 2.
        if avg == best_acc:
            image_model_state_dict = image_model.state_dict()
            image_model_state_dict = {key: image_model_state_dict[key].clone() for key in image_model_state_dict}
            text_model_state_dict = text_model.state_dict()
            text_model_state_dict = {key: text_model_state_dict[key].clone() for key in text_model_state_dict}

    image_model.load_state_dict(image_model_state_dict)
    text_model.load_state_dict(text_model_state_dict)
    test(0, is_eval=False)
    summary_writer.close()

def fx_calc_map_multilabel_k(retrieval, retrieval_labels, query, query_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(query, retrieval, metric)
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = dist.shape[1]
    res = []
    for i in range(numcases):
        order = ord[i].reshape(-1)[0: k]

        tmp_label = (np.dot(retrieval_labels[order], query_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)

if __name__ == '__main__':
    main()