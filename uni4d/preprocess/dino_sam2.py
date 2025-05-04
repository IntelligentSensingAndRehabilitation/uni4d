import os
import argparse
import cv2
import json
import torch
import imageio
import numpy as np
import tempfile
import supervision as sv
from pathlib import Path
from supervision.draw.color import ColorPalette
import sys

sys.path.insert(0, "../Grounded-SAM-2")

from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def id_to_colors(id):  # id to color
    rgb = np.zeros((3,), dtype=np.uint8)
    for i in range(3):
        rgb[i] = id % 256
        id = id // 256
    return rgb


def frame_generator(cap):
    """Generate frames from a VideoCapture object"""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def get_sam2_models(sam2_checkpoint, model_cfg, grounding_model):
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino from huggingface
    processor = AutoProcessor.from_pretrained(grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        grounding_model
    ).to(DEVICE)
    return sam2_predictor, grounding_model, processor


def run_sam2(cap, processor, sam2_predictor, grounding_model, text_input):
    all_obj_info = []
    mask_frames = []

    idx_to_id = [i for i in range(256 * 256 * 256)]
    np.random.shuffle(idx_to_id)  # mapping to randomize idx to id to get random color

    for frame_id, frame in tqdm(enumerate(frame_generator(cap))):
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sam2_predictor.set_image(image)

        inputs = processor(images=image, text=text_input, return_tensors="pt").to(
            DEVICE
        )
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        height, width = image.shape[:2]
        target_sizes = [(height, width)]
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.3,
            target_sizes=target_sizes,
        )

        # get the box prompt for SAM 2
        input_boxes = results[0]["boxes"].cpu().numpy()

        if input_boxes.shape[0] != 0:

            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            """
            Post-process the output of the model to get the masks, scores, and logits for visualization
            """
            # convert the shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            confidences = results[0]["scores"].cpu().numpy().tolist()
            class_names = results[0]["labels"]
            class_ids = np.array(list(range(len(class_names))))

            """
            Visualize image with supervision useful API
            """
            detections = sv.Detections(
                xyxy=input_boxes,  # (n, 4)
                mask=masks.astype(bool),  # (n, h, w)
                class_id=class_ids,
                confidence=np.array(confidences),
            )

            assert len(detections.class_id) > 0

            nms_idx = (
                torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy).float(),
                    torch.from_numpy(detections.confidence).float(),
                    0.5,
                )
                .numpy()
                .tolist()
            )

            detections.xyxy = detections.xyxy[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.mask = detections.mask[nms_idx]

            masks = detections.mask
            labels = detections.class_id

            assert (
                np.sum(labels == -1) == 0
            )  # check if any label == -1? concept graph has a bug with this

            color_mask = np.zeros(image.shape, dtype=np.uint8)

            obj_info_json = []

            # sort masks according to size
            mask_size = [np.sum(mask) for mask in masks]
            sorted_mask_idx = np.argsort(mask_size)[::-1]

            for idx in sorted_mask_idx:  # render from largest to smallest

                mask = masks[idx]
                color_mask[mask] = id_to_colors(idx_to_id[idx])

                obj_info_json.append(
                    {
                        "id": idx_to_id[idx],
                        "label": class_names[labels[idx]],
                        "score": float(detections.confidence[idx]),
                    }
                )

            mask_frames.append(color_mask)
            # Store JSON in list instead of file
            all_obj_info.append(obj_info_json)

        else:
            # Empty frame with no detections
            empty_mask = np.zeros(image.shape, dtype=np.uint8)

            mask_frames.append(empty_mask)

            # Add empty list to JSON data
            all_obj_info.append([])

    mask_array = np.array(mask_frames, dtype=np.uint8)
    # save mask frames as npz
    mask_temp_file = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez(mask_temp_file.name, masks=mask_array)
    # Close the temporary file so it can be used later
    mask_temp_file.close()

    return mask_temp_file.name, all_obj_info
