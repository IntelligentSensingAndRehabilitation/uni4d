import datajoint as dj
import tempfile
import numpy as np
import os
import json
from pose_pipeline import Video


schema = dj.schema('uni4d')

@schema
class RamGptSettingsLookup(dj.Lookup):
    definition = """
    ram_gpt_settings_id: int unsigned
    ---
    ram_gpt_settings: varchar(256)
    """

    contents = [
        {'ram_gpt_settings_id': 1, 'ram_gpt_settings': '{"image_size":384, "model_path": "/home/jd/uni4d/preprocess/pretrained/ram_swin_large_14m.pth"}'},
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

        from uni4d.preprocess.ram_gpt import load_model, process_capture
        kwargs = json.loads((RamGptSettingsLookup & key).fetch1('ram_gpt_settings'))
        model = load_model(kwargs['model_path'], image_size=kwargs['image_size'])

        output = process_capture(capture, model, device="cuda")
        key['ram_gpt_output'] = json.dumps(output)
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
    """

    contents = [
        {'cotracker_settings_id': 1, 'interval': 20, 'grid_size': 25, 'model_path': '/home/jd/uni4d/preprocess/pretrained/scaled_offline.pth'},
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

        from uni4d.preprocess.cotracker import load_model, process_capture
        kwargs = (CoTrackerSettingsLookup & key).fetch1()
        model = load_model(kwargs['model_path'])

        output = process_capture(capture, model, interval=kwargs['interval'], grid_size=kwargs['grid_size'])
        # add output dict to key
        key.update({
            'tracks': output['tracks'],
            'visibilities': output['visibilities'],
            'confidences': output['confidences'],
            'init_frames': output['init_frames'],
            'orig_shape': output['orig_shape']
        })
        self.insert1(key)
        print(f"Processed {key} with CoTracker model.")
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
    """
    contents = [
        {'unidepth_settings_id': 1, 'use_gt_intrinsics': False, 'use_v2': False},
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

        from uni4d.preprocess.unidepth import load_model, run_unidepth
        kwargs = (UniDepthSettingsLookup & key).fetch1()
        model = load_model(use_v2=kwargs['use_v2'])

        output = run_unidepth(capture, model, use_gt_K=kwargs['use_gt_intrinsics'])
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
        np.save(tmp_file.name, output['depth'].astype('float32'))
        tmp_file.close()
        
        # add output dict to key
        key['intrinsics'] = output['intrinsics']
        
        # Insert with the file path - DataJoint will handle copying the file
        key['depth'] = tmp_file.name
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
    """

    contents = [
        {'dino_sam2_settings_id': 1,
        "grounding_model":"IDEA-Research/grounding-dino-base",
        "sam2_checkpoint":"./preprocess/pretrained/sam2.1_hiera_large.pt",
        "sam2_model_config":"configs/sam2.1/sam2.1_hiera_l.yaml",
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

        grounding_model, sam2_checkpoint, sam2_model_config = (DinoSam2SettingsLookup & key).fetch1(
            'grounding_model', 'sam2_checkpoint', 'sam2_model_config')
        
        cap = Video.get_robust_reader(key)
        predictor, grounding_model, processor = get_sam2_models(
            sam2_checkpoint, sam2_model_config, grounding_model)
        masks_path, obj_info = run_sam2(cap, processor, predictor, grounding_model, text_input)

        key['masks'] = masks_path
        key['obj_info'] = obj_info
        self.insert1(key)
        os.remove(masks_path)

if __name__ == "__main__":
    CoTracker.populate('filename LIKE "0502%"')


        

