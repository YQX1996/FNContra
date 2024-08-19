import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import lpips
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
from torch import nn

from train import generated_neg

import json
import torch.nn.functional as F
from torchvision import utils as vutils, models
from models import Generator, Discriminator
import argparse
import numpy as np
import torch
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from benchmarking.calc_inception import load_patched_inception_v3

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH

@torch.no_grad()
def extract_features(loader, inception):
    pbar = tqdm(loader)
    feature_list = []
    for img, _ in pbar:
        img = img.cuda()
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))
    features = torch.cat(feature_list, 0)
    return features

def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f'Imaginary component {m}')
        cov_sqrt = cov_sqrt.real
    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff
    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)
    fid = mean_norm + trace
    return fid

def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def resize(img):
    return F.interpolate(img, size=256)

def batch_generate(zs, netG, batch=8):
    g_images = []
    with torch.no_grad():
        for i in range(len(zs) // batch):
            g_images.append(netG(zs[i * batch:(i + 1) * batch]).cpu())
        if len(zs) % batch > 0:
            g_images.append(netG(zs[-(len(zs) % batch):]).cpu())
    return torch.cat(g_images)

def compute_kid(feat_real, feat_fake, num_subsets=100, max_subset_size=1000):
    n = feat_real.shape[1]
    m = min(min(feat_real.shape[0], feat_fake.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = feat_fake[np.random.choice(feat_fake.shape[0], m, replace=False)]
        y = feat_real[np.random.choice(feat_real.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return t / num_subsets / m


def batch_save(images, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for i, image in enumerate(images):
        vutils.save_image(image.add(1).mul(0.5), folder_name + '/%d.jpg' % i)


def normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(pretrained=True).features
        self.channels = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                self.channels.append(layer.out_channels)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)

def calculate_lpips_given_data_loader(data_loader):
    lpips_model = lpips.LPIPS(net='vgg').cuda()
    lpips_distances = []

    for batch in tqdm(data_loader, desc='Calculating LPIPS'):
        images, _ = batch  # Assuming your DataLoader returns images and their labels (which are ignored here)
        images = images.cuda()
        num_images = len(images)

        # Pairwise comparison of images
        for i in range(num_images):
            for j in range(i + 1, num_images):
                img1 = images[i]
                img2 = images[j]

                # Compute LPIPS distance
                distance = lpips_model(img1, img2, normalize=True)
                lpips_distances.append(distance.detach().cpu().numpy())

    # Calculate average LPIPS distance
    average_lpips = torch.mean(torch.tensor(lpips_distances))
    return json.dumps(average_lpips.cpu().numpy().tolist())

# 定义函数用于生成圆形分布的数据
def generate_circle_data(radius, num_samples):
    angle = np.linspace(0, 2*np.pi, num_samples)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return np.vstack((x, y)).T

# 对每组特征进行聚类，生成圆形分布的聚类结果
def cluster_and_visualize(features, label, color):
    # 将张量移动到 CPU 上
    features = features.cpu()

    # 使用 K-means 聚类对特征进行聚类
    num_clusters = 1
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(features)

    # 将特征降维到二维
    tsne = TSNE(n_components=2)
    transformed_features = tsne.fit_transform(features)

    # 可视化
    plt.scatter(transformed_features[:, 0], transformed_features[:, 1], c=color, label=label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate images')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--artifacts',
                        type=str,
                        default=r"./train_results/FNContra-new/all/256/100-shot-grumpy_cat",
                        help='path to artifacts.')
    parser.add_argument('--task',
                        type=str,
                        default="batch",
                        help='index of gpu to use')
    parser.add_argument('--start_iter', type=int, default=6)
    parser.add_argument('--end_iter', type=int, default=6)
    parser.add_argument('--dist', type=str, default='.')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=64, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=5000)
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--path_real', type=str,
                        default=r'./few-shot-images/100-shot-grumpy_cat')
    parser.add_argument('--path_fake', type=str, default='None')
    parser.set_defaults(big=False)
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.Resize((args.size, args.size)),
         # transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
         ]
    )
    print('path_real: ', args.path_real)
    dset_a = ImageFolder(args.path_real, transform)
    loader_a = DataLoader(dset_a, batch_size=args.batch, num_workers=4)

    noise_dim = 256
    net_ig = Generator(ngf=64, nz=noise_dim, nc=3, im_size=args.size)  # , big=args.big )
    net_ig.cuda()

    net_D = Discriminator(ndf=64, im_size=args.size)
    net_D.cuda()

    for epoch in [10000 * i for i in range(args.start_iter, args.end_iter + 1)]:
        ckpt = f"{args.artifacts}/models/{epoch}.pth"
        checkpoint = torch.load(ckpt, map_location=lambda a, b: a)
        checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
        checkpoint['d'] = {k.replace('module.', ''): v for k, v in checkpoint['d'].items()}
        net_ig.load_state_dict(checkpoint['g'])
        net_D.load_state_dict(checkpoint['d'])

        # net_ig.eval()
        print('load checkpoint success, epoch %d' % epoch)
        net_ig.cuda()
        net_D.cuda()
        del checkpoint

        dist_fake = args.artifacts + '_%s_%d' % (args.task, epoch)
        dist = os.path.join(dist_fake, 'img')
        os.makedirs(dist, exist_ok=True)

        with torch.no_grad():
            for i in tqdm(range(args.n_sample // args.batch)):
                noise = torch.randn(args.batch, noise_dim).cuda()
                g_imgs = (net_ig(noise)[0])
                # g_imgs_HF = dwt_init(g_imgs)[1] + dwt_init(g_imgs)[2] + dwt_init(g_imgs)[3]
                #
                # _, features_A, _ = net_D(g_imgs, 'fake')
                # neg_easy, neg_middle, neg_hard = generated_neg(g_imgs)
                # _, features_B, _ = net_D(neg_easy, 'fake')
                # _, features_C, _ = net_D(neg_middle, 'fake')
                # _, features_D, _ = net_D(neg_hard, 'fake')
                #
                # # 聚类并可视化每组特征
                # plt.figure(figsize=(10, 8))
                # cluster_and_visualize(features_A, 'real', 'red')
                # cluster_and_visualize(features_B, 'easy', 'blue')
                # cluster_and_visualize(features_C, 'hard', 'green')
                # cluster_and_visualize(features_D, 'ultra-hard', 'orange')
                #
                # plt.title('Moongate')
                # # plt.xlabel('t-SNE Dimension 1')
                # # plt.ylabel('t-SNE Dimension 2')
                # plt.legend()
                # plt.grid(True)
                # plt.show()
                #

                for j, g_img in enumerate(g_imgs):
                    vutils.save_image(g_img.add(1).mul(0.5), os.path.join(dist, '%d.png' % (i * args.batch + j)))

            inception = load_patched_inception_v3().eval().cuda()
            features_a = extract_features(loader_a, inception).numpy()
            print(f'extracted {features_a.shape[0]} features')

            real_mean = np.mean(features_a, 0)
            real_cov = np.cov(features_a, rowvar=False)

            dset_b = ImageFolder(dist_fake, transform)
            loader_b = DataLoader(dset_b, batch_size=args.batch, num_workers=4)

            features_b = extract_features(loader_b, inception).numpy()
            print(f'extracted {features_b.shape[0]} features')

            sample_mean = np.mean(features_b, 0)
            sample_cov = np.cov(features_b, rowvar=False)
            fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
            kid = compute_kid(features_a, features_b)
            LPIPS_value = calculate_lpips_given_data_loader(loader_b)

            # report FID values
            filename = os.path.join(args.artifacts, 'FID_%.5i_%s.json' % (epoch, args.task))
            save_json(fid, filename)

            # report KID values
            filename = os.path.join(args.artifacts, 'KID_%.5i_%s.json' % (epoch, args.task))
            save_json(kid, filename)

            # report LPIPS values
            filename = os.path.join(args.artifacts, 'LPIPS_%.5i_%s.json' % (epoch, args.task))
            save_json(LPIPS_value, filename)
