import torch
import imageio
import numpy as np
import glob
import os
from unidepth.models import UniDepthV2, UniDepthV2old
from unidepth.utils.camera import Pinhole, BatchCamera
import argparse
from tqdm import tqdm
import time
import cv2

def load_model(use_v2=False):

    if load_model.model is None:
        load_model.model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14") if use_v2 else UniDepthV2old.from_pretrained(f"lpiccinelli/unidepth-v2old-vitl14")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_model.model = load_model.model.to(device)
    return load_model.model

load_model.model = None

def run_unidepth(cap, model, use_gt_K=False):
    images = []

    if not cap.isOpened():
        print(f"Error: Could not open video")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB since cv2 reads in BGR format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(frame_rgb)
        
        cap.release()

    # Convert the list of images to a tensor
    images = torch.tensor(np.array(images))  # F x H x W x 3

    f, h, w, c = images.shape
    print(f'{f} frames, {h} height, {w} width, {c} channels')
    batch_size = 8

    if not use_gt_K:

        with torch.no_grad():
                                
            frame_range = list(range(f))
            chuncks = [frame_range[i:i+batch_size] for i in range(0, len(frame_range), batch_size)]
            initial_intrinsics = []
            for chunk in chuncks:
                imgs = images[chunk, ...]
                imgs = torch.permute(imgs, (0, 3, 1, 2))
                preds = model.infer(imgs)
                initial_intrinsics.append(preds['intrinsics'])

        initial_intrinsics = torch.cat(initial_intrinsics, dim=0)
        initial_intrinsics = torch.mean(initial_intrinsics, dim=0)

        K = initial_intrinsics.detach().cpu()

        K[0][-1] = w / 2
        K[1][-1] = h / 2

    else:
        raise NotImplementedError("Have not implemented ground truth intrinsics yet.")

    if len(K.shape) == 2:
        K = K.unsqueeze(0).repeat(images.shape[0], 1, 1)

    depths = []

    with torch.no_grad():
                        
        frame_range = list(range(f))
        chuncks = [frame_range[i:i+batch_size] for i in range(0, len(frame_range), batch_size)]
        for chunk in tqdm(chuncks):
            imgs = images[chunk, ...]
            Ks = K[chunk, ...]
            if isinstance(model, (UniDepthV2)):
                Ks = Pinhole(K=Ks)
                cam = BatchCamera.from_camera(Ks)
            imgs = torch.permute(imgs, (0, 3, 1, 2))
            preds = model.infer(imgs, Ks)
            depth = preds['depth']                                         # B x 1 x H x W
            depths.append(depth)

        depths = torch.cat(depths, dim=0)
        depths = depths.detach().cpu().numpy()
    
    if False:
        depth_vis = depths[:,0,:,:]
        disp_vis = 1/(depth_vis + 1e-12)
        disp_vis = cv2.normalize(disp_vis,  disp_vis, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        for f in range(len(disp_vis)):
            disp = disp_vis[f]
            disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)  
            imageio.imwrite(os.path.join(str(f).zfill(4)+".png"), disp)

    print(depths.shape)
    return {'depth':depths, 'intrinsics':K[0].detach().cpu().numpy()}

