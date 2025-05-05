import torch
torch.cuda.set_per_process_memory_fraction(1.0, 0)  # The 0 means no pre-allocation
import cv2
import datajoint as dj
import tempfile
import numpy as np
import os
import json
from pose_pipeline import Video


schema = dj.schema("uni4d")

class DownsampledCapture:
    def __init__(self, capture, downsample_factor):
        self.capture = capture
        self.downsample_factor = downsample_factor

    def read(self):
        ret, frame = self.capture.read()
        if ret:
            if self.downsample_factor > 1:
                h, w = frame.shape[:2]
                new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return ret, frame

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.capture.get(prop_id) / self.downsample_factor
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.capture.get(prop_id) / self.downsample_factor
        return self.capture.get(prop_id)

    def set(self, prop_id, value):
        return self.capture.set(prop_id, value)

    def release(self):
        self.capture.release()

    def __getattr__(self, name):
        """Forward any undefined attributes/methods to the underlying capture object"""
        return getattr(self.capture, name)

@schema
class RamGptSettingsLookup(dj.Lookup):
    definition = """
    ram_gpt_settings_id: int unsigned
    ---
    ram_gpt_settings: varchar(256)
    downsample: int unsigned
    """

    contents = [
        {
            "ram_gpt_settings_id": 1,
            "ram_gpt_settings": '{"image_size":384, "model_path": "/home/jd/uni4d/preprocess/pretrained/ram_swin_large_14m.pth"}',
            "downsample": 2,
        }
        # Add more settings as needed
    ]


@schema
class RamGpt(dj.Computed):
    definition = """
    # RamGPT model for 4D pose estimation
    -> Video
    -> RamGptSettingsLookup
    ---
    ram_gpt_output: longblob
    """

    def make(self, key):
        capture = Video.get_robust_reader(key)
        downsample_factor = (RamGptSettingsLookup & key).fetch1("downsample")
        capture = DownsampledCapture(capture, downsample_factor)

        from uni4d.preprocess.ram_gpt import load_model, process_capture

        kwargs = json.loads((RamGptSettingsLookup & key).fetch1("ram_gpt_settings"))
        model = load_model(kwargs["model_path"], image_size=kwargs["image_size"])

        output = process_capture(capture, model, device="cuda")
        key["ram_gpt_output"] = json.dumps(output)
        self.insert1(key)
        print(f"Processed {key} with RAM+GPT model.")
        capture.release()

    @property
    def key_source(self):
        # Return the source of keys for this table
        return (Video & 'filename NOT LIKE "%.%"') * RamGptSettingsLookup


@schema
class CoTrackerSettingsLookup(dj.Lookup):
    definition = """
    cotracker_settings_id: int unsigned
    ---
    interval: int unsigned
    grid_size: int unsigned
    model_path: varchar(256)  # Path to the CoTracker model checkpoint
    downsample: int unsigned
    """

    contents = [
        {
            "cotracker_settings_id": 1,
            "interval": 20,
            "grid_size": 25,
            "model_path": "/home/jd/uni4d/preprocess/pretrained/scaled_offline.pth",
            "downsample": 2,
        },
        # Add more settings as needed
    ]


@schema
class CoTracker(dj.Computed):
    definition = """
    # CoTracker model for 4D pose estimation
    -> Video
    -> CoTrackerSettingsLookup
    ---
    tracks: longblob
    visibilities: longblob
    confidences: longblob
    init_frames: longblob
    orig_shape: longblob  # Original shape of the video frames
    """

    def make(self, key):
        capture = Video.get_robust_reader(key)
        downsample_factor = (CoTrackerSettingsLookup & key).fetch1("downsample")
        capture = DownsampledCapture(capture, downsample_factor)

        from uni4d.preprocess.cotracker import load_model, process_capture

        kwargs = (CoTrackerSettingsLookup & key).fetch1()
        model = load_model(kwargs["model_path"])

        output = process_capture(
            capture, model, interval=kwargs["interval"], grid_size=kwargs["grid_size"]
        )
        # add output dict to key
        key.update(
            {
                "tracks": output["tracks"],
                "visibilities": output["visibilities"],
                "confidences": output["confidences"],
                "init_frames": output["init_frames"],
                "orig_shape": output["orig_shape"],
            }
        )
        self.insert1(key)
        print(f"Processed {key['filename']} with CoTracker model.")
        capture.release()

    @property
    def key_source(self):
        # Return the source of keys for this table
        return (Video & 'filename NOT LIKE "%.%"') * CoTrackerSettingsLookup


@schema
class UniDepthSettingsLookup(dj.Lookup):
    definition = """
    unidepth_settings_id: int unsigned
    ---
    use_gt_intrinsics: bool
    use_v2: bool
    downsample: int unsigned
    """
    contents = [
        {
            "unidepth_settings_id": 1,
            "use_gt_intrinsics": False,
            "use_v2": False,
            "downsample": 2,
        },
        # Add more settings as needed
    ]


@schema
class UniDepth(dj.Computed):
    definition = """
    # UniDepth model for depth estimation
    -> Video
    -> UniDepthSettingsLookup
    ---
    intrinsics: longblob
    depth: attach@localattach 
    """

    def make(self, key):
        capture = Video.get_robust_reader(key)
        downsample_factor = (UniDepthSettingsLookup & key).fetch1("downsample")
        capture = DownsampledCapture(capture, downsample_factor)

        from uni4d.preprocess.unidepth import load_model, run_unidepth

        kwargs = (UniDepthSettingsLookup & key).fetch1()
        model = load_model(use_v2=kwargs["use_v2"])

        output = run_unidepth(capture, model, use_gt_K=kwargs["use_gt_intrinsics"])
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
        np.save(tmp_file.name, output["depth"].astype("float32"))
        tmp_file.close()

        # add output dict to key
        key["intrinsics"] = output["intrinsics"]

        # Insert with the file path - DataJoint will handle copying the file
        key["depth"] = tmp_file.name
        self.insert1(key)
        os.remove(tmp_file.name)
        capture.release()

    @property
    def key_source(self):
        # Return the source of keys for this table
        return (Video & 'filename NOT LIKE "%.%"') * UniDepthSettingsLookup


@schema
class DinoSam2SettingsLookup(dj.Lookup):
    definition = """
    dino_sam2_settings_id: int unsigned
    ---
    sam2_checkpoint: varchar(256)
    sam2_model_config: varchar(256)
    grounding_model: varchar(256)
    downsample: int unsigned
    """

    contents = [
        {
            "dino_sam2_settings_id": 1,
            "sam2_checkpoint": "/home/jd/uni4d/preprocess/pretrained/sam2.1_hiera_large.pt",
            "sam2_model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "grounding_model": "IDEA-Research/grounding-dino-base",
            "downsample": 2,
        }
    ]


@schema
class DinoSam2(dj.Computed):
    definition = """
    -> DinoSam2SettingsLookup
    -> RamGpt
    ---
    masks: attach@localattach  # Path to the output masks
    obj_info: longblob
    """

    def make(self, key):
        from uni4d.preprocess.dino_sam2 import get_sam2_models, run_sam2

        dyn_objs = json.loads((RamGpt & key).fetch1("ram_gpt_output"))["dynamic"]
        text_input = ". ".join(dyn_objs) + "."

        grounding_model, sam2_checkpoint, sam2_model_config = (
            DinoSam2SettingsLookup & key
        ).fetch1("grounding_model", "sam2_checkpoint", "sam2_model_config")
        sam2_checkpoint = '/home/jd/projects/uni4d/preprocess/pretrained/sam2.1_hiera_large.pt'

        cap = Video.get_robust_reader(key)
        downsample_factor = (RamGptSettingsLookup & key).fetch1("downsample")
        cap = DownsampledCapture(cap, downsample_factor)

        predictor, grounding_model, processor = get_sam2_models(
            sam2_checkpoint, sam2_model_config, grounding_model
        )
        masks_path, obj_info = run_sam2(
            cap, processor, predictor, grounding_model, text_input
        )

        key["masks"] = masks_path
        key["obj_info"] = obj_info
        self.insert1(key)
        os.remove(masks_path)


@schema
class Deva(dj.Computed):
    definition = """
    -> DinoSam2
    ---
    annotations: attach@localattach # Path to the output annotations
    args: longblob
    pred: longblob
    """

    def make(self, key):
        from uni4d.preprocess.deva import (
            run_deva,
            create_deva_workspace,
            remove_deva_workspace,
        )

        # Create the workspace for Deva
        create_deva_workspace(key)

        # Run Deva on the created workspace
        mask_file, args, pred = run_deva(key)

        # Store the results in the key
        key["annotations"] = mask_file
        key["args"] = args
        key["pred"] = pred

        self.insert1(key)
        print(f"Processed {key['filename']} with Deva.")
        remove_deva_workspace(key)
        os.remove(mask_file)  # Clean up the mask file after insertion

@schema
class Uni4d(dj.Computed):
    definition = """
    -> RamGpt
    -> CoTracker
    -> UniDepth
    -> DinoSam2
    -> Deva
    ---
    c2w : longblob  # Camera-to-world transformation matrix
    intrinsics: longblob
    """

    class Fused4d(dj.Part):
        definition = """
        -> master
        ---
        fused_4d: attach@localattach  # Path to the fused 4D data 
        """

    def make(self, key, save_fused=True):
        from uni4d.preprocess.uni4d import run_uni4d, uni4d_to_datajoint, remove_uni4d_workspace, create_uni4d_workspace
        # Create the workspace for Uni4D
        create_uni4d_workspace(key)
        # Run Uni4D on the created workspace
        run_uni4d(key)
        # Convert the results to DataJoint format
        res = uni4d_to_datajoint(key)
        # Store the results in the key
        key["c2w"] = res["c2w"]
        key["intrinsics"] = res["intrinsics"]
        self.insert1(key)
        if save_fused:
            # Save the fused 4D data
            fused_4d_path = f'{key["filename"]}_uni4d_workspace/video1/uni4d/fused_4d.npz'
            self.Fused4D.insert1({"master": key, "fused_4d": fused_4d_path})
        print(f"Processed {key['filename']} with Uni4D.")
        # Remove the workspace after processing
        remove_uni4d_workspace(key)

if __name__ == "__main__":
    CoTracker.populate('filename LIKE "0502%" AND cotracker_settings_id = 2')
    CoTracker.populate('filename LIKE "0601%" AND cotracker_settings_id = 2')
    # Uni4d.populate()


