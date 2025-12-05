import math
import os
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from modules import xfeat
from modules.xfeat import XFeat
from modules.lighterglue import LighterGlue

from losses import NLLLoss
from omegaconf import OmegaConf

from tqdm import tqdm

from dataset import MatcherDataset

import tensorboard


def configurate_optimizer(matcher, max_lr, wd):
    optimizer = torch.optim.AdamW(
        matcher.parameters(),
        lr=max_lr,
        weight_decay=wd,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer

def configurate_scheduler(optimizer, max_lr, steps_per_epoch, epochs, pct_start, div_factor, final_div_factor):
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=pct_start,                  
        anneal_strategy='cos',          
        div_factor=div_factor,               
        final_div_factor=final_div_factor,      
    )
    return scheduler

def configure_dataloaders(device, dir, img_size, difficulty, batch_size, val_batch_size):
    train_dataset = MatcherDataset(device, dir, img_size, difficulty, mode="train")
    val_dataset   = MatcherDataset(device, dir, img_size, difficulty, mode="val")
    test_dataset  = MatcherDataset(device, dir, img_size, difficulty, mode="test")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader

def configurate_matcher(weights_path, device):
    matcher = LighterGlue(weights=weights_path).to(device).train()
    return matcher

def configurate_xfeat(weights_path, device):
        xfeat = XFeat(weights=weights_path).to(device).eval()
        for param in xfeat.parameters():
            param.requires_grad = False
        return xfeat

def train(epochs, matcher, xfeat, train_loader, val_loader, config, loss_func, optimizer, scheduler, device):
    
    def __compute_gt(xfeat_output_0, xfeat_output_1, warp, mask):
        keypoints_0, keypoints_1 = xfeat_output_0, xfeat_output_1 # [B, 4096, 2]
        H = warp # [3, 3]
        mask = mask # [B, H, W]

        pass #TODO

    def __collect_xfeat_to_batch(xfeat_output, W, H, B):
        kps = []
        dss = []
        sizes = []

        lengths = [sample['keypoints'].shape[0] for sample in xfeat_output]
        M = min(lengths)

        for sample in xfeat_output:
            k = sample['keypoints']
            d = sample['descriptors']

            k = k[:M]
            d = d[:M]

            kps.append(k)
            dss.append(d)
            sizes.append(torch.tensor([W1, H1], dtype=torch.float32, device=k.device))
        
        keypoints   = torch.stack(kps, dim=0)   # [B, M, 2]
        descriptors = torch.stack(dss, dim=0)   # [B, M, 64]
        image_size = torch.tensor([[W, H]] * B, dtype=torch.float32, device=img.device) # [B, 2(W, H)]

        return keypoints, descriptors, image_size

    max_norm = config['max_norm']
    W1 = W2 = config['img_size_w']
    H1 = H2 = config['img_size_h']
    B = config['batch_size']
    for epoch in range(epochs):
        matcher.train()
        for  img, res, (warp, mask) in tqdm(train_loader):
            img = img.to(device)
            res = res.to(device)

            with torch.no_grad():   
                xfeat_output_0 = xfeat.detectAndCompute(img, top_k=4096)
                xfeat_output_1 = xfeat.detectAndCompute(res, top_k=4096)
                
            keypoints_0, descriptors_0, image_size_0 = __collect_xfeat_to_batch(xfeat_output_0, W1, H1, B)
            keypoints_1, descriptors_1, image_size_1 = __collect_xfeat_to_batch(xfeat_output_1, W2, H2, B)

            gt = __compute_gt(( keypoints_0, descriptors_0, image_size_0), (keypoints_1, descriptors_1, image_size_1), warp, mask) 

            data = {
                'keypoints0': keypoints_0.to(device),
                'descriptors0': descriptors_0.to(device),
                'image_size0': image_size_0.to(device),

                'keypoints1': keypoints_1.to(device),
                'descriptors1': descriptors_1.to(device),
                'image_size1': image_size_1,
            }

            optimizer.zero_grad()

            prediction = matcher(data)

            nll, _, loss_dict = loss_func(prediction, gt)
            loss = nll.mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                matcher.parameters(), 
                max_norm=max_norm
            )

            optimizer.step()
            scheduler.step()

if __name__ == "__main__":
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        matcher = configurate_matcher(config['matcher_weights'], device)
        xfeat = configurate_xfeat(config['xfeat_weights'], device)

        train_loader, val_loader, test_loader = configure_dataloaders(
            device, config['dataset_dir'], (config['img_size_h'], config['img_size_w']),
            config['difficulty'], config['batch_size'], config['val_batch_size'])


        optimizer = configurate_optimizer(matcher, config['max_lr'], config['wd'])
        scheduler = configurate_scheduler(optimizer, config['max_lr'], len(train_loader),
            config['epochs'], config['pct_start'], config['div_factor'],
            config['final_div_factor'])

        loss_conf = OmegaConf.create({"nll_balancing": 0.5, "gamma_f": 0.0})
        criterion = NLLLoss(loss_conf)