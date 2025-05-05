import numpy as np
import os
import subprocess
import cv2
from pose_pipeline import Video
import json
from PIL import Image
from uni4d.datajoint.uni4d_dj import (
    DinoSam2,
    DinoSam2SettingsLookup,
    DownsampledCapture,
)


def run_deva(key):
    """Run Deva on the given key. This was really annoying to implement as a standalone function, so just running it liek this."""
    # resolve paths relative to the current file
    run_deva_path = os.path.join(
        os.path.dirname(__file__),
        "../../Tracking-Anything-with-DEVA/evaluation/eval_with_detections.py",
    )
    model_path = os.path.join(
        os.path.dirname(__file__), "../../preprocess/pretrained/DEVA-propagation.pth"
    )
    # run deva
    subprocess.run(
        [
            "python",
            run_deva_path,
            "--workdir",
            f"{key['filename']}_deva_workspace",  # workspace dir is local
            "--model",
            model_path,
        ]
    )

    # convert to datajoint format
    mask_file, args, pred = deva_workspace_to_datajoint(key)
    return mask_file, args, pred


def create_deva_workspace(key):
    """Modifying Deva to not work from the command line is too much work"""
    os.makedirs(f'{key["filename"]}_deva_workspace', exist_ok=True)
    os.makedirs(f'{key["filename"]}_deva_workspace/video1', exist_ok=True)
    os.makedirs(f'{key["filename"]}_deva_workspace/video1/rgb', exist_ok=True)
    os.makedirs(f'{key["filename"]}_deva_workspace/video1/gsam2', exist_ok=True)
    os.makedirs(f'{key["filename"]}_deva_workspace/video1/gsam2/mask', exist_ok=True)

    mask, json_list, downsample_factor = (
        DinoSam2 * DinoSam2SettingsLookup & key
    ).fetch1("masks", "obj_info", "downsample")

    # write every frame of mask to video1/gsam2/mask/XXXX.png (mask is an np.ndarray of shape (N, H, W, 3))
    mask = np.load(mask)["masks"]

    for frame_id, mask_frame in enumerate(mask):
        pil_img = Image.fromarray(mask_frame)
        pil_img.save(
            f'{key["filename"]}_deva_workspace/video1/gsam2/mask/{frame_id:05d}.png'
        )

    # write json to video1/gsam2/mask/XXXX.json
    for json_id, json_data in enumerate(json_list):
        with open(
            f'{key["filename"]}_deva_workspace/video1/gsam2/mask/{json_id:05d}.json',
            "w",
        ) as f:
            json.dump(json_data, f)

    # get video
    video_key = (Video & key).fetch1("KEY")
    video_reader = Video.get_robust_reader(video_key)
    video_reader = DownsampledCapture(video_reader, downsample_factor=downsample_factor)
    # write every frame of video to video1/rgb/XXXX.png
    frame_id = 0
    while True:
        ret, frame = video_reader.read()
        if not ret:
            break
        # Convert BGR to RGB before saving
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_img.save(f'{key["filename"]}_deva_workspace/video1/rgb/{frame_id:05d}.jpg')
        # cv2.imwrite(f'{key["filename"]}_deva_workspace/video1/rgb/{frame_id:05d}.jpg', frame)
        frame_id += 1


def remove_deva_workspace(key):
    """Remove the Deva workspace for the given key"""
    workspace_path = f'{key["filename"]}_deva_workspace'
    if os.path.exists(workspace_path):
        import shutil

        shutil.rmtree(workspace_path)
        print(f"Removed workspace: {workspace_path}")
    else:
        print(f"Workspace does not exist: {workspace_path}")


def deva_workspace_to_datajoint(key):
    # load dictionary from args.txt file
    base_working_dir = f'{key["filename"]}_deva_workspace/video1/deva'
    import json
    import tempfile

    with open(os.path.join(base_working_dir, "args.txt"), "r") as f:
        args = json.load(f)
    with open(os.path.join(base_working_dir, "pred.json"), "r") as f:
        pred = json.load(f)

    # read every image file from video1/deva/Annotations/XXXX.png and put into a npz
    import numpy as np

    mask_dir = os.path.join(base_working_dir, "Annotations")
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    masks = []
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask_img = Image.open(mask_path)
        masks.append(np.array(mask_img))
    masks = np.array(masks)

    fname = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
    np.savez(fname.name, masks=masks)
    fname.close()
    return fname.name, args, pred
