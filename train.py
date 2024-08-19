import argparse
import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from tqdm import tqdm
from Wavelet_HH import WavePool_HH, generated_neg_D
from diffaug import DiffAugment
from models import weights_init, Generator, Discriminator
from operation import ImageFolder, InfiniteSamplerWrapper
from operation import copy_G_params, load_params, get_dir
import lpips_model

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
policy = 'color,translation'
percept = lpips_model.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)
extract_HH = WavePool_HH(3).cuda()
generated_neg = generated_neg_D(3).cuda()

def crop_image_by_part(image, part):
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label == "real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part], feat_pos, real_proto = net(data, label, part=part)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
              percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum() + \
              percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum() + \
              percept(rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2])).sum()

        return err, feat_pos, real_proto
    else:
        pred, _, _ = net(data, label)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()

        return err

def InfoNCE(anchor, positive, negative):
    temperature = 0.07

    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negative = F.normalize(negative, dim=1)

    pos_similarity = torch.sum(anchor * positive, dim=1) / temperature
    neg_similarity = torch.sum(anchor * negative, dim=1) / temperature

    logits = torch.cat((pos_similarity.unsqueeze(1), neg_similarity.unsqueeze(1)), dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def train(args):
    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = True
    dataloader_workers = 8
    current_iteration = args.start_iter
    saved_model_folder, saved_image_folder = get_dir(args)

    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
        transforms.Resize((int(im_size), int(im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    trans = transforms.Compose(transform_list)

    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers,
                                 pin_memory=True))

    # from model_s import Generator, Discriminator
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)
    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['g'].items()})
        netD.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['d'].items()})
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt

    for iteration in tqdm(range(current_iteration, total_iterations + 1)):
        real_image = next(dataloader)
        real_image = real_image.cuda()
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).cuda()
        fake_images = netG(noise)
        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

        trans = transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomRotation(30),
        ])
        real_img_aug = trans(real_image)

        ## 2. train Discriminator
        netD.zero_grad()

        err_dr, pos, real_proto = train_d(netD, real_image, label="real")
        err_df = train_d(netD, [fi.detach() for fi in fake_images], label="fake")

        '''contra-D'''
        neg_easy, neg_middle, neg_hard = generated_neg(real_image)
        with torch.no_grad():
            _, anchor, _ = netD(real_img_aug, 'fake')
            _, easy, easy_proto = netD(neg_easy, 'fake')
            _, middle, middle_proto = netD(neg_middle, 'fake')
            _, hard, hard_proto = netD(neg_hard, 'fake')

        # # sim_cos
        # sim_easy = 1 - torch.cosine_similarity(pos, easy, dim=1).mean()
        # sim_easy = torch.exp(-sim_easy)
        # sim_easy = -torch.log10(sim_easy)
        #
        # sim_middle = 1 - torch.cosine_similarity(pos, middle, dim=1).mean()
        # sim_middle = torch.exp(-sim_middle)
        # sim_middle = -torch.log10(sim_middle)
        #
        # sim_hard = 1 - torch.cosine_similarity(pos, hard, dim=1).mean()
        # sim_hard = torch.exp(-sim_hard)
        # sim_hard = -torch.log10(sim_hard)
        #
        # if iteration == 0:
        #     total_sim_easy = sim_easy
        #     total_sim_middle = sim_middle
        #     total_sim_hard = sim_hard
        # else:
        #     total_sim_easy = 0.99 * total_sim_easy + 0.01 * (sim_easy)
        #     total_sim_middle = 0.99 * total_sim_middle + 0.01 * (sim_middle)
        #     total_sim_hard = 0.99 * total_sim_hard + 0.01 * (sim_hard)

        total_sim_easy = 0.5
        total_sim_middle = 0.5
        total_sim_hard = 0.5

        contra_easy_loss = total_sim_easy * InfoNCE(pos, anchor.detach(), easy.detach())
        contra_middle_loss = total_sim_middle * InfoNCE(pos, anchor.detach(), middle.detach())
        contra_hard_loss = total_sim_hard * InfoNCE(pos, anchor.detach(), hard.detach())

        loss_contra_D = contra_easy_loss + contra_hard_loss + contra_middle_loss
        loss_d = err_df + err_dr + loss_contra_D
        loss_D = loss_d

        loss_D.backward()
        optimizerD.step()

        ## 3. train Generator
        netG.zero_grad()
        pred_g, gen, gen_proto = netD(fake_images, "fake")
        err_g = -pred_g.mean()

        '''proto'''
        total_proto_gen2real = torch.tensor(0)
        total_proto_gen2easy = torch.tensor(0)
        total_proto_gen2middle = torch.tensor(0)
        total_proto_gen2hard = torch.tensor(0)

        for i in range(0, len(gen_proto)):
            d_proto_gen2real = F.pairwise_distance(gen_proto[i], real_proto[i].detach()).mean()
            d_proto_gen2easy = F.pairwise_distance(gen_proto[i], easy_proto[i].detach()).mean()
            d_proto_gen2middle = F.pairwise_distance(gen_proto[i], middle_proto[i].detach()).mean()
            d_proto_gen2hard = F.pairwise_distance(gen_proto[i], hard_proto[i].detach()).mean()

            total_proto_gen2real = total_proto_gen2real + 1 / (2 ** i) * d_proto_gen2real
            total_proto_gen2easy = total_proto_gen2easy + 1 / (2 ** i) * d_proto_gen2easy
            total_proto_gen2middle = total_proto_gen2middle + 1 / (2 ** i) * d_proto_gen2middle
            total_proto_gen2hard = total_proto_gen2hard + 1 / (2 ** i) * d_proto_gen2hard

        loss_proto_easy = torch.max(torch.tensor(0).cuda(),
                                    total_proto_gen2real - total_proto_gen2easy + total_sim_easy)
        loss_proto_middle = torch.max(torch.tensor(0).cuda(),
                                      total_proto_gen2real - total_proto_gen2middle + total_sim_middle)
        loss_proto_hard = torch.max(torch.tensor(0).cuda(),
                                    total_proto_gen2real - total_proto_gen2hard + total_sim_hard)
        loss_proto = loss_proto_easy + loss_proto_middle + loss_proto_hard
        loss_proto = 0.1 * loss_proto

        loss_g = err_g + loss_proto
        loss_g.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 10 == 0:
            print("GAN: loss d: %.3f loss g: %.3f contraD: %.3f contraG: %.3f"
                  % (err_dr.item(), err_g.item(), loss_contra_D.item(), loss_proto.item()))

        if iteration % 1000 == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder
                                  + '/%d.jpg' % iteration, nrow=4)
            load_params(netG, backup_para)

        if iteration % 10000 == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g': netG.state_dict(),
                        'd': netD.state_dict()},
                       saved_model_folder + '/%d.pth' % iteration)
            load_params(netG, backup_para)
            torch.save({'g': netG.state_dict(),
                        'd': netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()},
                       saved_model_folder + '/all_%d.pth' % iteration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str,
                        default='./few-shot-images/skulls/img',
                        help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    # parser.add_argument('--cuda', type=int, default=2, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='skulls', help='experiment name')
    parser.add_argument('--iter', type=int, default=30000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None',
                        help='checkpoint weight path if have one')

    args = parser.parse_args()
    print(args)
    train(args)

