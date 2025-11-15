import os
import cv2
import argparse
import subprocess
from tqdm import tqdm
import numpy as np
import tempfile
import torch
from torchvision import transforms
from PIL import Image
import imageio.v3 as iio

to_tensor = transforms.ToTensor()

def read_image_folder(folder_path):
    image_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    frames = [to_tensor(Image.open(p).convert("RGB")) for p in image_files]
    return torch.stack(frames)

def img2video(subfolder_path, output_path, fps=8):
    img_tensor = read_image_folder(subfolder_path)
    if img_tensor is None:
        print(f"Failed to read images from {subfolder_path}")
        return
    img_tensor = img_tensor.permute(0, 2, 3, 1)  
    frames = (img_tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()  
    iio.imwrite(
        output_path,
        frames,
        fps=fps,
        codec='libx264rgb',
        pixelformat='bgr24',
        macro_block_size=None,
        ffmpeg_params=['-crf', '0'],
    )
    print(f"Video saved to {output_path}")

def resize_video(input_path, output_path, scale_factor=0.25):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Create a temporary directory to store frames
    with tempfile.TemporaryDirectory() as temp_dir:
        pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(input_path)}")
        frame_count = 0
        
        # Extract and resize frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize the frame
            resized_frame = cv2.resize(frame, (new_width, new_height),
                                      interpolation=cv2.INTER_LINEAR)
            
            # Save the frame (BGR to RGB for compatibility with PyTorch)
            resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frame_path = os.path.join(temp_dir, f"frame_{frame_count:08d}.png")
            cv2.imwrite(frame_path, resized_frame_rgb)
            
            frame_count += 1
            pbar.update(1)
        
        # Use img2video instead of FFmpeg
        pbar.set_description(f"Encoding {os.path.basename(output_path)} with img2video")
        
        # Call the img2video function
        img2video(temp_dir, output_path, fps=fps)
        
        cap.release()
        pbar.close()
        return True

def process_video_folder(input_folder, output_folder, scale_factor=0.25):
    print("Bilinear interpolation video resizing")
    print(f"Processing videos in {input_folder} and saving to {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = [f for f in os.listdir(input_folder) if os.path.splitext(f.lower())[1] in video_extensions]
    
    print(f"Found {len(video_files)} video files to process")
    
    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}{os.path.splitext(video_file)[1]}")
        
        print(f"Processing: {video_file}")
        resize_video(input_path, output_path, scale_factor)
        
    print("All videos processed successfully")

def main():
    parser = argparse.ArgumentParser(description='Resize videos using bilinear interpolation')
    parser.add_argument('--input', '-i', required=True, help='Input folder containing videos')
    parser.add_argument('--output', '-o', required=True, help='Output folder for resized videos')
    parser.add_argument('--scale', '-s', type=float, default=0.25, help='Scale factor (default: 0.25 = 1/4 size)')
    
    args = parser.parse_args()
    
    process_video_folder(args.input, args.output, args.scale)

if __name__ == "__main__":
    main()