import torch
torch.cuda.set_per_process_memory_fraction(1.0, 0)  # The 0 means no pre-allocation
import cv2
import os
# add to path
import sys
import imageio
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import argparse
from tqdm import tqdm

# sys.path.insert(0,'../../co-tracker')
this_file_dir = os.path.dirname(os.path.abspath(__file__))
co_tracker_path = os.path.join(this_file_dir, '../../co-tracker')
sys.path.insert(0, os.path.abspath(co_tracker_path))
from cotracker.predictor import CoTrackerPredictor 

def load_model(model_path):
    """
    Load the CoTracker model from the specified path.
    """
    print(f"Loading model from {model_path}...")
    model = CoTrackerPredictor(model_path)
    print("Model loaded successfully.")
    return model


def process_capture(capture, model, interval, grid_size):
    """
    Run the CoTracker model on the video capture.
    """

    # read the whole video

    frames = []

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    capture.release()

    # Convert to NumPy array: (F, H, W, C)
    video_np = np.stack(frames, axis=0)

    # Convert to PyTorch tensor: (F, H, W, C) → (F, C, H, W) → (1, F, C, H, W)
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()
    if torch.cuda.is_available():
        video = video.cuda()

    # Then:
    _, F, _, H, W = video.shape
    QUERY_FRAMES = [x for x in range(0, F, interval)]

    all_tracks = np.zeros((F, 0, 2))
    all_visibilities = np.zeros((F, 0))
    all_confidences = np.zeros((F, 0))
    all_init_frames = []

    if torch.cuda.is_available():
        model = model.cuda()

    for f in tqdm(QUERY_FRAMES):

        # run the model
        pred_tracks, pred_visibility, pred_confidence = model(video, grid_size=grid_size, grid_query_frame=f, backward_tracking=True)

        frame_tracks = pred_tracks[0].cpu().numpy()
        frame_visibilities = pred_visibility[0].cpu().numpy()
        frame_confidences = pred_confidence[0].cpu().numpy()

        init_frames = np.repeat(f, frame_tracks.shape[1])

        all_tracks = np.concatenate((all_tracks, frame_tracks), axis=1)
        all_visibilities = np.concatenate((all_visibilities, frame_visibilities), axis=1)
        all_confidences = np.concatenate((all_confidences, frame_confidences), axis=1)
        all_init_frames.extend(init_frames)

    # Convert to NumPy arrays
    all_tracks = all_tracks.astype(np.float32)
    all_visibilities = all_visibilities.astype(np.float32)
    all_confidences = all_confidences.astype(np.float32)
    all_init_frames = np.array(all_init_frames, dtype=np.int32)

    return {
        'tracks': all_tracks,
        'visibilities': all_visibilities,
        'confidences': all_confidences,
        'init_frames': all_init_frames,
        'orig_shape': np.array(video.shape[-2:])
    }

