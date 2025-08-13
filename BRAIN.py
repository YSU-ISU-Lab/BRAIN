seed = 123
import numpy as np
from sympy import arg

np.random.seed(seed)
import random as rn

rn.seed(seed)
import os

os.environ['PYTHONHASHSEED'] = str(seed)
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from utils.config import args
import random
import time
from datetime import datetime

import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

cudnn.benchmark = True

from utils.bar_show import progress_bar
import pdb
from src.cmdataset import CMDataset
import scipy
import scipy.spatial
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
device_ids = [0, 1]
teacher_device_id = [0, 1]
best_acc = 0  # best test accuracy
start_epoch = 0

args.log_dir = os.path.join(args.root_dir, 'logs', args.log_name)
args.ckpt_dir = os.path.join(args.root_dir, 'ckpt', args.pretrain_dir)

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)

# import sys
# sys.stdout = open('flickr-16.txt', 'w')

import snntorch as snn
from snntorch import surrogate

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

class SNNHashing(nn.Module):
    def __init__(self, input_dim, beta, layers, hash_dim=64):
        super(SNNHashing, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8192)
        self.fc2 = nn.Linear(8192, hash_dim)
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)
        self.layers = layers

    def forward(self, features, is_train):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        mem2_rec = []
        if is_train:
            for step in range(self.layers):
                cur1 = self.fc1(features[step])
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2_rec.append(spk2)
                hash_code = torch.tanh(mem2)
                norm_x = torch.norm(hash_code, dim=1, keepdim=True)
                hash_code = hash_code / norm_x
                mem2_rec.append(hash_code)
        else:
            for step in range(self.layers):
                cur1 = self.fc1(features)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2_rec.append(spk2)
                hash_code = torch.tanh(mem2)
                norm_x = torch.norm(hash_code, dim=1, keepdim=True)
                mem2_rec = hash_code / norm_x
        return spk2_rec, mem2_rec


class CrossModalHashingNetwork(nn.Module):
    def __init__(self, img_dim, txt_dim, common_dim, hash_dim, img_layers, txt_layers):
        super(CrossModalHashingNetwork, self).__init__()
        self.image_snn = SNNHashing(img_dim, 0.99, img_layers, hash_dim)
        self.text_snn = SNNHashing(txt_dim, 0.99, txt_layers, hash_dim)

    def forward(self, img_feat, txt_feat, is_train=True):
        image_spk, image_mem = self.image_snn(img_feat, is_train)
        text_spk, text_mem = self.text_snn(txt_feat, is_train)

        return image_spk, image_mem, text_spk, text_mem

def attraction_loss(hash_code1, hash_code2, margin=0.5):
    distance = torch.sqrt(((hash_code1 - hash_code2) ** 2).sum(dim=1) + 1e-8)  # Avoid sqrt(0)
    loss = torch.clamp(distance - margin, min=0).mean()
    return loss


def time_encoding(features, T=1.0):
    min_val = np.min(features, axis=1, keepdims=True)
    max_val = np.max(features, axis=1, keepdims=True)
    normalized_features = (features - min_val) / (max_val - min_val)
    spike_times = T * (1 - normalized_features)
    return spike_times


def generate_mask(data_shape, num_steps, rate):
    masks = []
    previous_mask = np.ones(data_shape)
    for step in range(num_steps):
        # 生成随机掩码
        current_mask = previous_mask * (np.random.rand(*data_shape) > rate).astype(float)
        masks.append(current_mask)
        previous_mask = current_mask

    return np.stack(masks, axis=0)


def apply_mask(data, masks):
    masked_data = []
    masked_data.append(data)
    for step in range(masks.shape[0]):
        masked_data.append(data * masks[step])
    return np.stack(masked_data[::-1], axis=0)

def relational_distill_loss(student_features, teacher_features, temperature=0.07):
    # 归一化
    student_norm = F.normalize(student_features, dim=1)
    teacher_norm = F.normalize(teacher_features, dim=1)

    sim_s = torch.matmul(student_norm, student_norm.T) / temperature
    sim_t = torch.matmul(teacher_norm, teacher_norm.T) / temperature

    loss = F.kl_div(F.log_softmax(sim_s, dim=1), F.softmax(sim_t, dim=1), reduction='batchmean')
    return loss

def soft_info_nce_dual(student_feat1, student_feat2, teacher_feat1, teacher_feat2, temperature=0.1):
    # Normalize
    s1 = F.normalize(student_feat1, dim=1)
    s2 = F.normalize(student_feat2, dim=1)
    t1 = F.normalize(teacher_feat1, dim=1)
    t2 = F.normalize(teacher_feat2, dim=1)

    sim_s_12 = torch.matmul(s1, s2.T) / temperature
    sim_t_12 = torch.matmul(t1, t2.T) / temperature

    sim_s_21 = torch.matmul(s2, s1.T) / temperature
    sim_t_21 = torch.matmul(t2, t1.T) / temperature

    loss_12 = F.kl_div(F.log_softmax(sim_s_12, dim=1), F.softmax(sim_t_12, dim=1), reduction='batchmean')
    loss_21 = F.kl_div(F.log_softmax(sim_s_21, dim=1), F.softmax(sim_t_21, dim=1), reduction='batchmean')

    return 0.5 * (loss_12 + loss_21)
    
def main():
    print('===> Preparing data ..')
    # build data
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

    print('===> Building ResNet..')
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
    ckpt = torch.load(os.path.join(args.ckpt_dir, args.resume))
    image_model.load_state_dict(ckpt['image_model_state_dict'])
    text_model.load_state_dict(ckpt['text_model_state_dict'])
    image_model.eval()
    text_model.eval()
    
    model = CrossModalHashingNetwork(train_dataset.imgs.shape[1], train_dataset.text_dim, 8192, args.bit, int(args.time_steps[0]), int(args.time_steps[1])).cuda()
    parameters = list(model.parameters())
    wd = args.wd
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=wd)
    lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [300, 600, 900, 1200], gamma=0.1)
    summary_writer = SummaryWriter(args.log_dir)
    
    def train(epoch):
        print('\nEpoch: %d / %d' % (epoch, args.max_epochs))
        model.train()
        train_loss, correct, total = 0., 0., 0.
        for batch_idx, (idx, images, texts, _) in enumerate(train_loader):
            images, texts, idx = [img.cuda() for img in images], [txt.cuda() for txt in texts], [idx.cuda()]

            images_outputs_tea = [image_model(im) for im in images]
            texts_outputs_tea = [text_model(txt.float()) for txt in texts]
            
            image_masks = generate_mask(images[0].shape, int(args.time_steps[2]), args.time_enc1)
            text_masks = generate_mask(texts[0].shape, int(args.time_steps[3]), args.time_enc2)
            
            masked_image_features = apply_mask(images[0].cpu(), image_masks)
            masked_text_features = apply_mask(texts[0].float().cpu(), text_masks)
            masked_image_features = torch.from_numpy(masked_image_features).float().cuda()
            masked_text_features = torch.from_numpy(masked_text_features).float().cuda()

            image_spk, image_mem, text_spk, text_mem = model(masked_image_features, masked_text_features)

            image_spk_all = torch.stack(image_spk, dim=0)
            text_spk_all = torch.stack(text_spk, dim=0)
            
            image_spike_reg = image_spk_all.mean()
            text_spike_reg = text_spk_all.mean()
            
            loss_inter = relational_distill_loss(image_mem[-1], (images_outputs_tea[0]+texts_outputs_tea[0])/2, temperature=args.margin1) + \
                      relational_distill_loss(text_mem[-1], (images_outputs_tea[0]+texts_outputs_tea[0])/2, temperature=args.margin2)
            loss_intra = soft_info_nce_dual(image_mem[-1], text_mem[-1], images_outputs_tea[0], texts_outputs_tea[0], temperature=args.margin3)

            loss = 0.499 * loss_inter + 0.499 * loss_intra + 0.002 * (image_spike_reg + text_spike_reg)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            clip_grad_norm(parameters, 1.)
            optimizer.step()
            train_loss += loss.item()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | LR: %g'
                         % (train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))

            if batch_idx % args.log_interval == 0:  # every log_interval mini_batches...
                summary_writer.add_scalar('Loss/train', train_loss / (batch_idx + 1),
                                          epoch * len(train_loader) + batch_idx)
                summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'],
                                          epoch * len(train_loader) + batch_idx)

        print(
            f"loss: {loss.item()}")

    def eval(data_loader):
        imgs, txts, labs = [], [], []
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, texts, targets) in enumerate(data_loader):
                images, texts, targets = [img.cuda() for img in images], [txt.cuda() for txt in texts], targets.cuda()
                image_spk, image_mem, text_spk, text_mem = model(images[0], texts[0].float(), False)
                image_hash_code = [image_mem]
                text_hash_code = [text_mem]
                imgs += image_hash_code
                txts += text_hash_code
                labs.append(targets)

            imgs = torch.cat(imgs).sign_().cpu().numpy()
            txts = torch.cat(txts).sign_().cpu().numpy()
            labs = torch.cat(labs).cpu().numpy()
        return imgs, txts, labs

    import time
    def test(epoch, is_eval=True):
        # pass
        global best_acc
        # set_eval()
        model.eval()
        # switch to evaluate mode
        (retrieval_imgs, retrieval_txts, retrieval_labs) = eval(retrieval_loader)
        if is_eval:
            query_imgs, query_txts, query_labs = retrieval_imgs[0: 2000], retrieval_txts[0: 2000], retrieval_labs[
                                                                                                   0: 2000]
            retrieval_imgs, retrieval_txts, retrieval_labs = retrieval_imgs[0: 2000], retrieval_txts[
                                                                                      0: 2000], retrieval_labs[0: 2000]
        else:
            (query_imgs, query_txts, query_labs) = eval(query_loader)

        retrieval_txts_len = len(retrieval_txts)
        retrieval_imgs_len = len(retrieval_imgs)

        i2t = fx_calc_map_multilabel_k(retrieval_txts, retrieval_labs, query_imgs, query_labs, k=0, metric='hamming')
        t2i = fx_calc_map_multilabel_k(retrieval_imgs, retrieval_labs, query_txts, query_labs, k=0, metric='hamming')

        avg = (i2t + t2i) / 2.
        print('%s\nImg2Txt: %g \t Txt2Img: %g \t Avg: %g' % (
            'Evaluation' if is_eval else 'Test', i2t, t2i, (i2t + t2i) / 2.))
        if avg > best_acc:
            print('Saving..')
            # state = {
            #     'image_model_state_dict': image_model.state_dict(),
            #     'text_model_state_dict': text_model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'Avg': avg,
            #     'Img2Txt': i2t,
            #     'Txt2Img': t2i,
            #     'epoch': epoch,
            # }
            # torch.save(state, os.path.join(args.ckpt_dir, '%s_%d_best_checkpoint.t7' % (args.arch, args.bit)))
            # save_path = os.path.join(args.ckpt_dir, '%s_%d_best_test.t7' % (args.arch, args.bit))
            # checkpoint = {
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'Avg': avg,
            #     'Img2Txt': i2t,
            #     'Txt2Img': t2i,
            #     'epoch': epoch
            # }
            # torch.save(checkpoint, save_path)
            # torch.save(model.state_dict(), save_path)
            best_acc = avg
        return i2t, t2i

    lr_schedu.step(start_epoch)
    best_epoch = 0
    for epoch in range(start_epoch, args.max_epochs):
        train(epoch)
        lr_schedu.step(epoch)
        i2t, t2i = test(epoch)#, is_eval=False)
        avg = (i2t + t2i) / 2.
        if avg == best_acc:
            model_state_dict = model.state_dict()
            model_state_dict = {key: model_state_dict[key].clone() for key in model_state_dict}

    model.load_state_dict(model_state_dict)
    print('Test\n')
    test(0, is_eval=False)
    sys.stdout.close()
    sys.stdout = sys.__stdout__
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