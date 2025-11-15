import sys
import os
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np
import torch
import tqdm 
from tqdm import tqdm
import argparse
from raft import RAFT
from utils.utils import InputPadder
import warp_utils
import torch.nn.functional as F

# Adapted from https://github.com/phoenix104104/fast_blind_video_consistency
def compute_video_warping_error(video_path, model, device):

    cap = cv2.VideoCapture(video_path)
    frames = []
    warping_error = 0
    err = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)
        frames.append(frame)
    
    # Num = 5
    Num = len(frames)
    tensor_frames = torch.stack([torch.from_numpy(frame) for frame in frames])
    # for i in range(Num):
    N = len(tensor_frames)
    indices = torch.linspace(0, N - 1, Num).long()
    extracted_frames = torch.index_select(tensor_frames, 0, indices)
    with torch.no_grad():
        for i in range(Num - 1):
            frame1 = extracted_frames[i]
            frame2 = extracted_frames[i + 1]

            # Calculate optical flow using Farneback method
            img1 = frame1.permute(2,0,1).float().unsqueeze(0).to(device)/ 255.0
            img2 = frame2.permute(2,0,1).float().unsqueeze(0).to(device)/ 255.0
            # img1 = torch.tensor(img2tensor(frame1)).float().to(device)
            # img2 = torch.tensor(img2tensor(frame2)).float().to(device)

            # Downsample the images by a factor of 2
            img1 = F.interpolate(img1, scale_factor=0.5, mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, scale_factor=0.5, mode='bilinear', align_corners=False)

            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)

            ### compute fw flow
            
            _, fw_flow = model(img1, img2, iters=20, test_mode=True) # with optical flow model: RAFT
            fw_flow = warp_utils.tensor2img(fw_flow)
            # Clear cache and temporary data
            torch.cuda.empty_cache()

            ### compute bw flow
            _, bw_flow = model(img2, img1, iters=20, test_mode=True) # with optical flow model: RAFT
            bw_flow = warp_utils.tensor2img(bw_flow)
            torch.cuda.empty_cache()

            ### compute occlusion
            fw_occ, warp_img2 = warp_utils.detect_occlusion(bw_flow, fw_flow, img2)
            warp_img2 = torch.tensor(warp_img2).float().to(device)
            fw_occ = torch.tensor(fw_occ).float().to(device)

            ### load flow
            flow = fw_flow

            ### load occlusion mask
            occ_mask = fw_occ
            noc_mask = 1 - occ_mask

            # ipdb.set_trace()   
            diff = (warp_img2- img1) * noc_mask
            diff_squared = diff ** 2
            
            # Calculate the sum and mean
            N = torch.sum(noc_mask)
            if N == 0:
                N = diff_squared.numel()
            # ipdb.set_trace()
            err += torch.sum(diff_squared) / N

    warping_error = err / (len(extracted_frames) - 1)

    return warping_error

def Ewarp(args):
    pred = args.pred
    metric = args.metric

   
    video_paths = [os.path.join(pred, x) for x in os.listdir(pred)]
    from natsort import natsorted
    video_paths = natsorted([os.path.join(pred, x) for x in os.listdir(pred)])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if metric == 'warping_error':
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(device)
        model.eval()
        model.args.mixed_precision = False

    scores = []
    results = {}
    test_num = len(video_paths)
    count = 0
    for i in tqdm(range(len(video_paths))):
        video_path = video_paths[i]
        if count == test_num:
            break
        else:
            if metric == 'warping_error':
                basename = os.path.basename(video_path)
                score = compute_video_warping_error(video_path, model, device)
            score = score.item()
            score = score * 1000
            if score is not None and not np.isnan(score):
                scores.append(score)
                results[os.path.splitext(basename)[0]] = score
                count+=1
                average_score = sum(scores) / len(scores)
                print(f"Vid: {os.path.basename(video_path)},  Current {metric}: {score}, Current avg. {metric}: {average_score} ")
            
    average_score = sum(scores) / len(scores)
    print(f"Final average {metric}: {average_score}, Total videos: {len(scores)}")
    return results, average_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default='', help="Specify the path of generated videos")
    parser.add_argument("--metric", type=str, default='warping_error', help="Specify the metric to be used")
    parser.add_argument('--model', type=str, default='models/raft-things.pth',help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    Ewarp(args)