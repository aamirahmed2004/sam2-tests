import argparse
import os
import time
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2_video_predictor 
import cv2
import time 
from skimage.measure import label

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("Using CPU, not recommended.")

# For efficiency
if device.type == "cuda":
    torch.autocast("cuda", dtype = torch.float16).__enter__()

def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC

def apply_mask(image, mask, obj_id=None, image_size=None):
    """
    Applies a mask on the image and returns the masked image as a NumPy array.

    Args:
        image (PIL.Image or np.array): The original image.
        mask (np.array): The mask array (2D or 3D array).
        obj_id (int): ID used for mask color.
        image_size (tuple): The size to resize the mask.
        mask_bg (bool): Whether to apply the mask as a background.
        morph (bool): Whether to apply opening to get rid of noise followed by another round of dilation to slightly expand the mask.

    Returns:
        np.array: The masked image.
    """
    # Ensure image is in NumPy format
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    height, width = image.shape[:2]

    # Squeeze mask to handle extraneous dimensions
    mask = np.squeeze(mask)  # Removes singleton dimensions

    # Check dimensions after squeezing
    if len(mask.shape) != 2:
        raise ValueError(f"Unexpected mask shape: {mask.shape}. Expected 2D mask.")

    # Resize mask to match the image size
    if image_size is not None:
        mask = Image.fromarray((mask.astype(np.uint8) * 255))
        mask = mask.resize(image_size, resample=Image.NEAREST)
        mask = np.array(mask) > 128  # Convert back to boolean mask

    mask_uint8 = (mask.astype(np.uint8)) * 255

    # Kernels for morphological operations
    open_kernel = np.ones((3, 3), np.uint8)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Apply morphological opening to remove small noise (erosion followed by dilation)
    mask_open = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, open_kernel)
    mask = mask_open > 128
    
    if np.sum(mask) > 0:    # if there is at least one masked region
        largest_cc = getLargestCC(mask)
        # Check if the area of the largest component is at least 20% of the image area. If not, set the entire mask to 0's
        if np.sum(largest_cc) >= 0.2 * (height * width):
            mask = largest_cc
        else:
            mask = np.zeros_like(mask, dtype=bool)
    else:
        mask = np.zeros_like(mask, dtype=bool)

    # Dilate the mask after keeping only the largest connected component
    mask_dilated = cv2.dilate(mask.astype(np.uint8) * 255, dilation_kernel, iterations=2)
    mask = mask_dilated > 128

    # Mask the image itself i.e. remove the background 
    masked_image = np.zeros_like(image, dtype=np.uint8)
    for c in range(3):
        masked_image[:, :, c] = np.where(mask, image[:, :, c], 0)

    return masked_image

def save_processed_images(frames_path, frame_names, video_segments, frame_stride=1, output_dir="sam2_outputs"):

    print("Saving processed images")
    num_skipped_frames = 0
    start_time = time.time()

    for out_frame_idx in range(0, len(frame_names), frame_stride):
        image_path = os.path.join(frames_path, frame_names[out_frame_idx])
        image = Image.open(image_path)
        image_size = image.size  # Ensure masks match this size

        masked_image = np.array(image)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():  # this loop only runs once because there is only one obj id.
            masked_image = apply_mask(masked_image, out_mask, obj_id=out_obj_id, image_size=image_size)

        # Check if masked_image is all zeros
        if np.any(masked_image):
            output_path = os.path.join(output_dir, f'frame{out_frame_idx}.jpg')
            Image.fromarray(masked_image).save(output_path, "JPEG")
        else:
            num_skipped_frames += 1
        
    print(f"Time taken for saving images: {time.time() - start_time:.2f} seconds.")
    print(f"Number of frames skipped: {num_skipped_frames}")
    return num_skipped_frames

# Parse input arguments
parser = argparse.ArgumentParser(description="Testing Offloading Options for SAM2")
parser.add_argument("--input_path", type=str, help="Complete path to directory containing tracklet subdirectories")
args = parser.parse_args()

# ASSUMES YOU ARE IN SAM2-TESTS ROOT DIRECTORY
BASE_PATH = os.getcwd()
CHKPT_PATH = os.path.join(BASE_PATH, "checkpoints", "sam2.1_hiera_large.pt")
CONFIG_PATH = os.path.join(BASE_PATH, "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml")

INPUT_DIR = args.input_path
MASKED_PARENT_DIR = INPUT_DIR + "_masked"

if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError(f"Input directory {INPUT_DIR} does not exist.")

if not os.path.exists(MASKED_PARENT_DIR):
    os.mkdir(MASKED_PARENT_DIR)

tracklet_dirs = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]

predictor = build_sam2_video_predictor(CONFIG_PATH, CHKPT_PATH, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Dictionary to hold time taken to process (in seconds) and number of frames skipped, as a tuple
tracklet_info = {}
total_start_time = time.time()
inference_state = None

for tracklet in tracklet_dirs:

    if inference_state is not None:
        predictor.reset_state(inference_state)

    tracklet_path = os.path.join(INPUT_DIR, tracklet)
    output_dir = os.path.join(MASKED_PARENT_DIR, tracklet)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    frame_names = [p for p in os.listdir(tracklet_path) if os.path.splitext(p)[-1].lower() == ".jpg"]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"Before processing {tracklet}: GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    # Reset inference state for this tracklet and start timer
    start_time = time.time()
    inference_state = predictor.init_state(video_path=tracklet_path,
                                           offload_video_to_cpu=True,
                                           async_loading_frames=True)
    
    # Use the first frame to assign a bounding box for the entire image.
    image = Image.open(os.path.join(tracklet_path, frame_names[0]))
    width, height = image.size
    box = np.array([[5, 5], [width - 6, height - 6]])
    
    ann_frame_idx = 0
    ann_obj_id = 1  # arbitrary object ID
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box,
    )
    
    video_segments = {}
    # Process each frame in the video
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"After processing {tracklet}: GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    num_skipped = save_processed_images(tracklet_path, frame_names, video_segments, frame_stride=1, output_dir=output_dir)
    
    # Record and report time taken for this tracklet
    end_time = time.time()
    elapsed = end_time - start_time
    tracklet_info[tracklet] = (elapsed, num_skipped)
    print(f"Processed tracklet {tracklet} in {elapsed:.2f} seconds.")

total_end_time = time.time()
total_elapsed = total_end_time - total_start_time

# Write the timing info into a text file
results_file = os.path.join(MASKED_PARENT_DIR, "masking_run_info.txt")
with open(results_file, "w") as f:
    # Write each tracklet's timing info separated by commas
    for tracklet, t in tracklet_info.items():
        f.write(f"{tracklet},{t[0]:.2f},{t[1]}\n")
    # Add two newlines and then the total time taken
    f.write("\n\n")
    f.write(f"Total time taken: {total_elapsed:.2f} seconds")

print(f"All tracklets in {INPUT_DIR} processed. Total time: {total_elapsed:.2f} seconds. Results saved to {results_file}.")