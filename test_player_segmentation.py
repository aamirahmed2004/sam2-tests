"""
This code is adapted from the demo notebooks in the original sam2 repo, as well as tutorial at https://www.youtube.com/watch?v=Mf5w-cr2T8U. You can follow instructions there but here is a summary.

If you want to run run this code, you need to have the following installed: CUDA Toolkit 12.4 or above, PyTorch, matplotlib. 
Make sure CUDA Toolkit is added to the path. I believe you also need to install Visual Studio for the C/C++ dev kit for Windows that comes with it, which is needed to compile some CUDA code. A virtual environment is recommended for the other dependencies.

First, ensure that you download sam2.1_heira_large.pt from 
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt and place it in sam2/checkpoints/

Second, run "pip install -e ." (inside the sam2 repo)

Lastly, uncomment the call to display_annotation_frame() to see if the setup is complete

Important to consider: processing around 100 frames takes up 3GB of VRAM and upto 30 seconds. This approach is not fast, but if we pre-compute the masks, I believe the results will be much better for tracking the target player than Centroid ReID from the original paper.
"""

import os
import torch 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
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
    torch.autocast("cuda", dtype = torch.bfloat16).__enter__()

# Change this to the directory where you cloned this repo
BASE_PATH = "C:\\Users\\syeda\\OneDrive\\Desktop\\4th Year\\COSC419\\sam2-tests"
CHKPT_PATH = os.path.join(BASE_PATH, "checkpoints", "sam2.1_hiera_large.pt")
CONFIG_PATH = os.path.join(BASE_PATH, "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml")

def show_mask(mask, ax, obj_id=None, image_size=None, mask_bg=False):

    # Chosing a color
    cmap = plt.get_cmap("tab10")
    cmap_idx = 0 if obj_id is None else obj_id
    color = np.array([*cmap(cmap_idx)[:3], 0.6])

    # Remove extra 3rd dimension, mask should only be 2D
    mask = np.squeeze(mask)

    if image_size is not None:
        # Convert boolean mask to uint8 (0 or 255)
        mask_uint8 = (mask.astype(np.uint8)) * 255
        # Create a PIL image and resize it using nearest neighbor interpolation
        mask_pil = Image.fromarray(mask_uint8)
        mask_resized = mask_pil.resize(image_size, resample=Image.NEAREST)
        # Convert resized mask back to a boolean array
        mask = np.array(mask_resized) > 128

    assert(len(mask.shape) == 2)
    h, w = mask.shape

    if mask_bg:
        # Create an image with the mask applied, everything else black
        mask_image = np.zeros((h, w, 3), dtype=np.float32)
        mask_image[mask] = color[:3]
    else:
        # Create an overlay image by reshaping the mask and multiplying by the color
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    ax.imshow(mask_image)

def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC

def apply_mask(image, mask, obj_id=None, image_size=None, mask_bg=False, morph=False):
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

    # Generate color for the mask
    cmap = plt.get_cmap("tab10")
    color = np.array([*cmap(obj_id if obj_id is not None else 0)[:3]]) * 255  # RGB

    if morph:
        mask_uint8 = (mask.astype(np.uint8)) * 255

        # Kernels for morphological operations
        open_kernel = np.ones((3, 3), np.uint8)
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

        # Apply morphological opening to remove small noise (erosion followed by dilation)
        mask_open = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, open_kernel)
        # Dilate the mask after opening to expand the masked regions slightly
        mask_dilated = cv2.dilate(mask_open, dilation_kernel, iterations=1)

        mask = mask_dilated > 128
    
    if np.sum(mask) > 0:    # if there is at least one masked region
        largest_cc = getLargestCC(mask)
        # Check if the area of the largest component is at least 20% of the image area. If not, set the entire mask to 0's
        if np.sum(largest_cc) >= 0.2 * (height * width):
            mask = largest_cc
        else:
            mask = np.zeros_like(mask, dtype=bool)
    else:
        mask = np.zeros_like(mask, dtype=bool)

    if mask_bg:
        # Mask the image itself i.e. remove the background 
        masked_image = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            masked_image[:, :, c] = np.where(mask, image[:, :, c], 0)
    else:
        # Overlay mask with color on the image
        masked_image = image.copy()
        alpha = 0.6
        for c in range(3):
            masked_image[:, :, c] = np.where(mask, 
                                             (1 - alpha) * image[:, :, c] + alpha * color[c], 
                                             image[:, :, c])

    return masked_image

def show_chosen_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def display_initial_frame(frames_path, frame_names, frame_idx=0):
    """
    This was used only to display the frame with matplotlib's axes so I could manually find 2 points on the target player.
    """
    plt.figure(figsize=(12, 8))
    plt.title(f"Frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(frames_path, frame_names[frame_idx])))
    plt.show()

    print("Press key to continue")
    input()
    print("Continuing")

def display_annotation_frame(frames_path, frame_names, ann_frame_idx, points, labels, out_mask_logits, out_obj_ids):
    """
    This was used to display the initial mask returned by the predictor as a sanity check, before running the entire video through the model.
    
    There might be misalignment in overlaid mask, but I could not be bothered to fix this. If the shape is roughly correct, the sanity check passes and it means your setup is complete. You can then comment out this function being called. 
    """
    plt.figure(figsize=(12, 8))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(frames_path, frame_names[ann_frame_idx])))
    show_chosen_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    plt.show()

    print("Press key to continue")
    input()
    print("Continuing")

def save_processed_images(frames_path, frame_names, video_segments, frame_stride=1, output_dir="sam2_outputs", morph_mask=False):

    print("Saving processed images")
    num_skipped_frames = 0
    start_time = time.time()
    plt.close("all")

    final_output_dir = os.path.join(BASE_PATH, output_dir, INPUT_DIR)
    if not os.path.exists(final_output_dir):
        os.mkdir(final_output_dir)

    # Add counter before filename so that each outputs from run appear separately when browsing in the folder
    counter = 0
    output_path = os.path.join(final_output_dir, f'run{counter}_frame0.png')
    while os.path.exists(output_path):
        # Crude way of handling clashing filename so I don't overwrite outputs from different test runs 
        counter += 1
        output_path = os.path.join(final_output_dir, f'run{counter}_frame0.png')  

    for out_frame_idx in range(0, len(frame_names), frame_stride):
        image_path = os.path.join(frames_path, frame_names[out_frame_idx])
        image = Image.open(image_path)
        image_size = image.size  # Ensure masks match this size

        masked_image = np.array(image)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():  # this loop only runs once because there is only one obj id.
            masked_image = apply_mask(masked_image, out_mask, obj_id=out_obj_id, image_size=image_size, mask_bg=True, morph=morph_mask)

        # Check if masked_image is all zeros
        if np.any(masked_image):
            output_path = os.path.join(final_output_dir, f'run{counter}_frame{out_frame_idx}.jpg')
            Image.fromarray(masked_image).save(output_path, "JPEG")
        else:
            num_skipped_frames += 1
        
    print(f"Time taken for saving images: {time.time() - start_time:.2f} seconds.")
    print(f"Number of frames skipped: {num_skipped_frames}")
    return num_skipped_frames

def overlay_box(box, ax):   
    """
    Expects bbox in x_min,y_min,x_max,y_max format, and converts to x,y,w,h for plt.Rectangle
    """
    x1, y1 = box[0]
    x2, y2 = box[1]
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)

def get_max_height_and_width(frames_path):
    widths = []
    heights = []
    
    # Iterate over files in the directory
    for filename in os.listdir(frames_path):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(frames_path, filename)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    if not widths or not heights:
        print("No images processed successfully.")
        return
    
    return np.max(heights), np.max(widths)

def resize_and_pad_images(input_dir):

    output_dir = input_dir + "_resized"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    max_height, max_width = get_max_height_and_width(input_dir)
    pad_amounts = None
    
    for idx, filename in enumerate(os.listdir(input_dir)):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(input_dir, filename)
            try:
                with Image.open(img_path) as img:
                    # Create a new image with the max dimensions and a black background
                    new_img = Image.new("RGB", (max_width, max_height), (0, 0, 0))
                    # Calculate padding amounts
                    pad_horiz = (max_width - img.width) // 2
                    pad_vert = (max_height - img.height) // 2
                    # Paste the original image onto the center of the new image
                    new_img.paste(img, (pad_horiz, pad_vert))
                    # Save the new image to the output directory
                    new_img.save(os.path.join(output_dir, filename))
                    
                    # Track pad amounts for the first image only
                    if idx == 0:
                        pad_amounts = (pad_horiz, pad_vert)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
    
    return pad_amounts

parser = argparse.ArgumentParser(description="Testing Offloading Options for SAM2")
parser.add_argument("--visualize", action="store_true", help="Visualize bbox of first frame before running the model")
parser.add_argument("--offload_video", action="store_true", help="Offload video to CPU")
parser.add_argument("--offload_state", action="store_true", help="Offload state to CPU")
# parser.add_argument("--resize", action="store_true", help="Resize the frames in input frames directory to max height and width")
parser.add_argument("--morph", action="store_true", help="Applies opening followed by dilation on the output masks")
parser.add_argument("--frames_dir", type=str, default="sample_frames", help="Directory containing the frames to input to the model")
args = parser.parse_args()

INPUT_DIR = args.frames_dir

if args.offload_video and args.offload_state:
    OUTPUT_DIR = "offload_both_outputs"
    print("Offloading state and video to CPU")
elif args.offload_video:
    OUTPUT_DIR = "offload_video_outputs"
    print("Offloading video to CPU")
elif args.offload_state:
    OUTPUT_DIR = "offload_state_outputs"
    print("Offloading state to CPU")
else:
    OUTPUT_DIR = "sam2_outputs"
    print("Loading inference state and video frames to GPU memory")

OUTPUT_DIR = "mask_QA_outputs"

# Change to absolute path
OUTPUT_DIR = os.path.join(BASE_PATH, OUTPUT_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

frames_path = os.path.join(BASE_PATH, INPUT_DIR)
frame_names = [
    p for p in os.listdir(frames_path)
    if os.path.splitext(p)[-1] == ".jpg"
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

torch.cuda.empty_cache()  # remove lingering allocations
predictor = build_sam2_video_predictor(CONFIG_PATH, CHKPT_PATH, device=device)

model_allocated = torch.cuda.memory_allocated(device=device)
model_reserved = torch.cuda.memory_reserved(device=device)

print(f"Memory after loading model:")
print(f"  Allocated: {model_allocated / (1024 ** 3):.2f} GB")
print(f"  Reserved: {model_reserved / (1024 ** 3):.2f} GB")

# Don't need to display initial frame after we choose the points 
# display_initial_frame(frames_path, frame_names)

inference_state = predictor.init_state(video_path=frames_path, offload_video_to_cpu=args.offload_video, offload_state_to_cpu=args.offload_state, async_loading_frames=True)
predictor.reset_state(inference_state)

ann_frame_idx = 0   
ann_obj_id = 1      # this is the ID for the player, we could have chosen any integer

# Assign bounding box to the entire image
image = Image.open(os.path.join(frames_path, frame_names[0]))
width, height = image.size
labels = np.array([1,1], np.int32)  # this tells the predictor that the points in the previous line correspond to the same target object

# if args.resize:
#     box = np.array([[pad_horiz+3, pad_vert+3], [width - (pad_horiz+4), height - (pad_vert+4)]])
# else:
box = np.array([[5, 5], [width-6, height-6]])

# Display the initial frame with the bounding box overlay
if args.visualize:
    plt.figure(figsize=(12, 8))
    plt.title(f"Frame {ann_frame_idx}")
    image = Image.open(os.path.join(frames_path, frame_names[ann_frame_idx]))
    plt.imshow(image)
    overlay_box(box, plt.gca())
    plt.show()
    input()

_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state = inference_state,
    frame_idx = ann_frame_idx,
    obj_id = ann_obj_id,
    # points = points,
    box = box,
    # labels = labels
)

# This sanity check was only needed when setting up the code
# display_annotation_frame(frames_path, frame_names, ann_frame_idx, box, labels, out_mask_logits, out_obj_ids)

if device.type == "cuda":
    allocated = torch.cuda.memory_allocated(device=device) / (1024 ** 3)  
    reserved = torch.cuda.memory_reserved(device=device) / (1024 ** 3)  
    print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

video_segments = {} 
for out_frame_idx , out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):

    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

    
save_processed_images(frames_path, frame_names, video_segments, frame_stride=1, output_dir=OUTPUT_DIR)

"""
TODO:
    [X] 1) Fix the issue of the complete tracklet not fitting into VRAM; in that case split the video into two, and input the mask of the last frame of the first half into the second half
    [X] 2) use predictor.add_new_points_or_box(), and see if the results using a bounding box as prompt are as good as using 2 points as a prompt (since bounding boxes can be automatically extracted from a lightweight model like YOLO). 
    [X] 3) if the output masks are good, then see if YOLOv11 generates good bounding boxes for a few sample tracklets

For testing bbox: here is a manually annotated bbox for frame 0, to pass into SAM2. This will simulate getting an automatic bounding box with the first player
    "x": "27.55",
    "y": "69.90",
    "width": "42.04",
    "height": "127.14"

    [X] 4) Figure out what format sam2 expects bounding boxes to be in. 
        Answer: (x_min,y_min,x_max,y_max)
    
    [x] 5) Fix problem with bounding box inputs not producing good masks.
    [x] 6) Experiment with different image sizes
    [x] 7) Try whether padding all images to be the same size (max height x max width) improves the output
    [x] 8) Come up with algorithm for determining whether key player was NOT in the first frame
    [x] 9) Automate YOLO bbox -> SAM2 pipeline
    [] 10) Figure out a method of eliminating frames based on a) size of image relative to average size, b) size of mask, c) movement?.
-----------------------
Experimenting to find the time taken and memory consumption to run the model while offloading state to CPU and/or offloading video to CPU

Tests conducted with 112 frames in /sample_frames: 
    No offloading: 2+28 seconds, ~2GB allocated
    Offload videos: 2+29 seconds, 400MB allocated 
    Offload state: 2+28 seconds, ~1.7GB allocated
    Offload both: 2+33 seconds, 400MB allocated

Tests conducted with 360 frames in /sample_frames2 (copied SoccerNet data train/2 directory):
    No offloading: takes WAY too long, ~5GB allocated
    Offload both: 9+113 seconds, 400MB allocated
    Offload video: 9+123, 400MB allocated

Summary: actual outputs are identical whether we offload or not. Probably worth offloading the video.
-----------------------
Experimenting with async_loading_frames=True results in better masks somehow, and basically eliminates the loading time (saving around 5 sec/100 frames)
-----------------------
Experimenting with changing image_size in sam2/configs/sam2.1/sam2.1_hiera_l.yaml to see if it takes less time

Tests with image size = 512, async loading, and offloading video to CPU.
    Total time: 20 seconds

Tests with image size = 384 (128*3), async loading, and offloading to CPU.
    Total time: 18 seconds, but IT DID NOT DETECT ANY MASKS

Tests with image size = 640 (128*5), async loading, and offloading to CPU.
    Total time: 36 seconds

-----------------------
Experimenting with padding the frames to max_height and max_width of all the frames in the tracklet since SAM2 expects uniformly sized frames for a video. 

Did not work because it thinks our pseudo-bounding boxes are telling it to keep track of the entire frame including background. Output masks are much worse and have more noise.
"""