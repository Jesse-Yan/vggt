import torch
import os
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

base_path = "dataset/v2x_vit"
mode = "train"
scenario = "2021_08_18_21_38_28"
vehicle_ids = ["8786"]
# vehicle_ids = ["8786", "8795"]
timestamps = ["000068", "000070"] # Changed timestamp to a list of timestamps
cameras = ["camera0", "camera1", "camera2", "camera3"]
camera_enabled = {
    "camera0": True,
    "camera1": True,
    "camera2": True,
    "camera3": True
}

mode_path = os.path.join(base_path, mode)

if scenario is None:
    scenario_folders = [f for f in os.listdir(mode_path) if os.path.isdir(os.path.join(mode_path, f))]
    scenario = scenario_folders[0]
scenario_path = os.path.join(mode_path, scenario)

image_names = [] # Initialize list before the loop
# Loop through each timestamp to collect image paths
for timestamp in timestamps:
    for vehicle_id in vehicle_ids:
        vehicle_path = os.path.join(scenario_path, vehicle_id)
        for camera in cameras:
            if camera_enabled[camera]:
                # Use the current timestamp from the loop
                image_path = os.path.join(vehicle_path, f"{timestamp}_{camera}.png")
                image_names.append(image_path)

# Process the combined list after the loop
print(image_names)

# Original processing logic now runs on the combined list
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        print("Extrinsic matrices:")
        print(extrinsic.shape)
        print(extrinsic)