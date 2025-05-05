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

class BoundedCapture:
    """
    A video capture wrapper that enforces frame boundaries and can be downsampled.
    Acts as a transparent standalone capture for a section of video.
    """
    def __init__(self, capture, start_frame, end_frame, downsample_factor=1):
        """
        Args:
            capture: OpenCV VideoCapture object
            start_frame: First frame in this section
            end_frame: Last frame in this section (exclusive)
            downsample_factor: Factor to downsample frames
        """
        self.capture = capture
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.downsample_factor = downsample_factor
        self.current_frame = start_frame
        
        # Position the capture at the start frame
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
    def read(self):
        """Read next frame, respecting section boundaries"""
        if self.current_frame >= self.end_frame:
            return False, None
            
        ret, frame = self.capture.read()
        if ret:
            self.current_frame += 1
            if self.downsample_factor > 1:
                h, w = frame.shape[:2]
                new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return ret, frame
    
    def get(self, prop_id):
        """Get property, adjusted for section and downsampling"""
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.end_frame - self.start_frame
        elif prop_id == cv2.CAP_PROP_POS_FRAMES:
            return self.current_frame - self.start_frame
        elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.capture.get(prop_id) / self.downsample_factor
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.capture.get(prop_id) / self.downsample_factor
        return self.capture.get(prop_id)
    
    def set(self, prop_id, value):
        """Set property, adjusted for section boundaries"""
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            # Adjust position relative to section start
            self.current_frame = min(self.start_frame + value, self.end_frame)
            return self.capture.set(prop_id, self.current_frame)
        return self.capture.set(prop_id, value)
    
    def release(self):
        """Release the underlying capture"""
        self.capture.release()
    
    def isOpened(self):
        """Check if capture is open and within section bounds"""
        return self.capture.isOpened() and self.current_frame < self.end_frame
    
    def __getattr__(self, name):
        """Forward any undefined attributes/methods to the underlying capture object"""
        return getattr(self.capture, name)


class SectionedCapture:
    """
    Divides a video into sections with configurable length and overlap.
    Creates BoundedCapture objects for each section.
    """
    def __init__(self, key, downsample_factor=1, section_length=90, section_overlap=10):
        self.key = key
        self.downsample_factor = downsample_factor
        self.section_length = section_length
        self.section_overlap = section_overlap
        
        # Create a temporary capture to get total frames
        temp_capture = Video.get_robust_reader(key)
        self.total_frames = int(temp_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_capture.release()
    
    def get_sections(self):
        """
        Returns a list of BoundedCapture objects, each representing one section.
        Each section gets its own independent capture object.
        """
        sections = []
        
        # Calculate start positions for each section
        start_frames = list(range(0, self.total_frames, self.section_length - self.section_overlap))
        
        # Make sure we don't start a section too close to the end
        start_frames = [f for f in start_frames if f + self.section_length // 2 <= self.total_frames]
        
        if not start_frames:
            # If video is too short, just use one section for the whole video
            start_frames = [0]
        
        # Store all captures to release them later
        self._captures = []
        
        for start_frame in start_frames:
            # Create a NEW capture for each section instead of sharing one
            capture = Video.get_robust_reader(self.key)
            self._captures.append(capture)
            
            # Determine end frame (either section_length frames later or end of video)
            end_frame = min(start_frame + self.section_length, self.total_frames)
            
            # Create a bounded capture for this section with its own capture object
            bounded_capture = BoundedCapture(
                capture=capture,
                start_frame=start_frame,
                end_frame=end_frame,
                downsample_factor=self.downsample_factor
            )
            
            sections.append(bounded_capture)
        
        return sections
    
    def release(self):
        """Release all underlying captures"""
        if hasattr(self, '_captures'):
            for capture in self._captures:
                capture.release()


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
    section_length: int unsigned  # Length of each section in frames
    section_overlap: int unsigned  # Overlap between sections in frames
    """

    contents = [
        {
            "cotracker_settings_id": 1,
            "interval": 20,
            "grid_size": 25,
            "model_path": "/home/jd/projects/uni4d/preprocess/pretrained/scaled_offline.pth",
            "downsample": 2,
            "section_length": 90,  # Length of each section in frames
            "section_overlap": 10,  # Overlap between sections in frames
        },
        # Add more settings as needed
    ]


@schema
class CoTracker(dj.Computed):
    definition = """
    # CoTracker model for 4D pose estimation
    -> Video
    -> CoTrackerSettingsLookup
    """

    class Section(dj.Part):
        definition = """
        -> master
        start_frame: int unsigned  # Start frame of the section
        end_frame: int unsigned  # End frame of the section
        ---
        tracks: longblob
        visibilities: longblob
        confidences: longblob
        init_frames: longblob
        orig_shape: longblob  # Original shape of the video frames
    """

    def make(self, key):
        # capture = Video.get_robust_reader(key)
        downsample_factor, section_length, section_overlap = (CoTrackerSettingsLookup & key).fetch1("downsample", "section_length", "section_overlap")
        sectioned = SectionedCapture(key, downsample_factor=downsample_factor, section_length=section_length, section_overlap=section_overlap)#.get_sectioned_captures()
        captures = sectioned.get_sections()

        from uni4d.preprocess.cotracker import load_model, process_capture

        kwargs = (CoTrackerSettingsLookup & key).fetch1()
        model = load_model(kwargs["model_path"])

        self.insert1(key)  # Insert the main key first
        for capture in captures:
            start_frame = capture.start_frame
            end_frame = capture.end_frame
            # print capture frame count
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Processing section from {start_frame} to {end_frame} with {frame_count} frames.")
            base_key = key.copy()
            output = process_capture(
                capture, model, interval=kwargs["interval"], grid_size=kwargs["grid_size"]
            )
            # add output dict to key
            base_key.update(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "tracks": output["tracks"],
                    "visibilities": output["visibilities"],
                    "confidences": output["confidences"],
                    "init_frames": output["init_frames"],
                    "orig_shape": output["orig_shape"],
                }
            )
            self.Section.insert1(base_key)
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
    -> CoTracker.Section
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
        base_key = key.copy()
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
            fused_4d_path = f'{key["filename"]}_{key["start_frame"]}_{key["end_frame"]}_uni4d_workspace/video1/uni4d/demo/fused_4d.npz'
            self.Fused4d.insert1({**base_key, "fused_4d": fused_4d_path})
        print(f"Processed {key['filename']} with Uni4D.")
        # Remove the workspace after processing
        remove_uni4d_workspace(key)

if __name__ == "__main__":
    CoTracker.populate('filename LIKE "0502_20230222%" AND cotracker_settings_id=2 AND video_project = "GAIT_CONTROLS"',reserve_jobs=True,suppress_errors=True)
    CoTracker.populate('filename LIKE "0601%" AND cotracker_settings_id = 1 AND video_project = "GAIT_CONTROLS"', reserve_jobs=True, suppress_errors=True)
    # Uni4d.populate()


