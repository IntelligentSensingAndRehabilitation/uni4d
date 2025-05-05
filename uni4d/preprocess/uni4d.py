import cv2
import subprocess
import numpy as np
import os
from PIL import Image
import json


def create_uni4d_workspace(key):
    from uni4d.datajoint.uni4d_dj import (
        RamGpt,
        CoTracker,
        UniDepth,
        DinoSam2,
        DinoSam2SettingsLookup,
        Deva,
        RamGptSettingsLookup,
        CoTrackerSettingsLookup,
        UniDepthSettingsLookup,
        DownsampledCapture,
        Deva,
    )
    from pose_pipeline import Video

    """Modifying uni4d to not work from the command line is too much work"""
    os.makedirs(f'{key["filename"]}_uni4d_workspace', exist_ok=True)
    os.makedirs(f'{key["filename"]}_uni4d_workspace/video1', exist_ok=True)
    os.makedirs(f'{key["filename"]}_uni4d_workspace/video1/rgb', exist_ok=True)
    os.makedirs(f'{key["filename"]}_uni4d_workspace/video1/gsam2', exist_ok=True)
    os.makedirs(f'{key["filename"]}_uni4d_workspace/video1/gsam2/mask', exist_ok=True)
    os.makedirs(f'{key["filename"]}_uni4d_workspace/video1/cotracker', exist_ok=True)
    os.makedirs(f'{key["filename"]}_uni4d_workspace/video1/deva', exist_ok=True)
    os.makedirs(
        f'{key["filename"]}_uni4d_workspace/video1/deva/Annotations', exist_ok=True
    )
    os.makedirs(f'{key["filename"]}_uni4d_workspace/video1/ram', exist_ok=True)
    os.makedirs(f'{key["filename"]}_uni4d_workspace/video1/unidepth', exist_ok=True)

    # fetch and save ramgpt data
    ram_output, ram_downsample = (RamGpt * RamGptSettingsLookup & key).fetch1(
        "ram_gpt_output", "downsample"
    )
    ram_output = json.loads(ram_output)  # ram_gpt_output is a JSON string
    ram_fname = f'{key["filename"]}_uni4d_workspace/video1/ram/tags.json'
    json.dump(ram_output, open(ram_fname, "w"), indent=4)

    # fetch and save cotracker data
    (
        cotracker_tracks,
        cotracker_visibilities,
        cotracker_confidences,
        cotracker_init_frames,
        cotracker_orig_shape,
        cotracker_downsample,
    ) = (CoTracker * CoTrackerSettingsLookup & key).fetch1(
        "tracks",
        "visibilities",
        "confidences",
        "init_frames",
        "orig_shape",
        "downsample",
    )
    cotracker_fname = f'{key["filename"]}_uni4d_workspace/video1/cotracker/results.npz'
    np.savez(
        cotracker_fname,
        all_confidences=cotracker_confidences,
        all_tracks=cotracker_tracks,
        all_visibilities=cotracker_visibilities,
        init_frames=cotracker_init_frames,
        orig_shape=cotracker_orig_shape,
    )
    # TODO: may have to do the filtered_results.npz as well

    # fetch and save unidepth data
    uni_depth, unidepth_intrinsics, unidepth_downsample = (
        UniDepth * UniDepthSettingsLookup & key
    ).fetch1("depth", "intrinsics", "downsample")
    uni_depth = np.load(uni_depth)
    uni_depth_fname = f'{key["filename"]}_uni4d_workspace/video1/unidepth/depth.npy'
    np.save(uni_depth_fname, uni_depth)
    intrinsics_fname = (
        f'{key["filename"]}_uni4d_workspace/video1/unidepth/intrinsics.npy'
    )
    np.save(intrinsics_fname, unidepth_intrinsics)

    # region gsam2 data
    mask, json_list, gsam2_downsample_factor = (
        DinoSam2 * DinoSam2SettingsLookup & key
    ).fetch1("masks", "obj_info", "downsample")

    # write every frame of mask to video1/gsam2/mask/XXXX.png (mask is an np.ndarray of shape (N, H, W, 3))
    mask = np.load(mask)["masks"]

    for frame_id, mask_frame in enumerate(mask):
        pil_img = Image.fromarray(mask_frame)
        pil_img.save(
            f'{key["filename"]}_uni4d_workspace/video1/gsam2/mask/{frame_id:05d}.png'
        )

    # write json to video1/gsam2/mask/XXXX.json
    for json_id, json_data in enumerate(json_list):
        with open(
            f'{key["filename"]}_uni4d_workspace/video1/gsam2/mask/{json_id:05d}.json',
            "w",
        ) as f:
            json.dump(json_data, f)

    # assert all downsample factors are the same
    assert (
        ram_downsample
        == cotracker_downsample
        == unidepth_downsample
        == gsam2_downsample_factor
    ), "Downsample factors do not match"

    # deva
    deva_annotations, deva_args, deva_pred = (Deva & key).fetch1(
        "annotations", "args", "pred"
    )
    masks = np.load(deva_annotations)["masks"]
    # save in deva/Annotations/XXXX.png
    for frame_id, mask_frame in enumerate(masks):
        pil_img = Image.fromarray(mask_frame)
        pil_img.save(
            f'{key["filename"]}_uni4d_workspace/video1/deva/Annotations/{frame_id:05d}.png'
        )

    # rgb video
    video_key = (Video & key).fetch1("KEY")
    video_reader = Video.get_robust_reader(video_key)
    video_reader = DownsampledCapture(video_reader, downsample_factor=ram_downsample)
    # write every frame of video to video1/rgb/XXXX.png
    frame_id = 0
    while True:
        ret, frame = video_reader.read()
        if not ret:
            break
        # Convert BGR to RGB before saving
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_img.save(f'{key["filename"]}_uni4d_workspace/video1/rgb/{frame_id:05d}.jpg')
        frame_id += 1

def remove_uni4d_workspace(key):
    """
    Remove the uni4d workspace for the given key.
    """
    workspace_path = f'{key["filename"]}_uni4d_workspace'
    if os.path.exists(workspace_path):
        import shutil
        shutil.rmtree(workspace_path)
        print(f"Removed workspace: {workspace_path}")
    else:
        print(f"Workspace does not exist: {workspace_path}")

def uni4d_to_datajoint(key):
    fused_4d_path = f'{key["filename"]}_uni4d_workspace/video1/uni4d/fused_4d.npz'
    if not os.path.exists(fused_4d_path):
        raise FileNotFoundError(f"Fused 4D data not found at {fused_4d_path}")
    
    fused_4d = np.load(fused_4d_path, allow_pickle=True)
    c2w = fused_4d["c2w"]
    intrinsices = fused_4d["Ks"]

    return {"c2w": c2w, "intrinsics": intrinsices}

def run_uni4d(key):
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../config/config_demo.yaml")
    )
    run_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../run.py"))
    command = [
        "python", run_path,
        "--gpu", "0",
        "--workdir", f"{key['filename']}_uni4d_workspace",
        "--config", config_path,
    ]
    try:
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
        print("Uni4D processing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Uni4D processing: {e}")
        return str(e)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return str(e)