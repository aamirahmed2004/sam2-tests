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
from sam2.build_sam import build_sam2_video_predictor # move this file into cloned sam2 repo to resolve this import error

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

def show_mask(mask, ax, obj_id=None, image_size=None):

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
    # Create an overlay image by reshaping the mask and multiplying by the color
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_chosen_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def display_initial_frame(frames_dir, frame_names, frame_idx=0):
    """
    This was used only to display the frame with matplotlib's axes so I could manually find 2 points on the target player.
    """
    plt.figure(figsize=(12, 8))
    plt.title(f"Frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(frames_dir, frame_names[frame_idx])))
    plt.show()

    print("Press key to continue")
    input()
    print("Continuing")

def display_annotation_frame(frames_dir, frame_names, ann_frame_idx, points, labels, out_mask_logits, out_obj_ids):
    """
    This was used to display the initial mask returned by the predictor as a sanity check, before running the entire video through the model.
    
    There might be misalignment in overlaid mask, but I could not be bothered to fix this. If the shape is roughly correct, the sanity check passes and it means your setup is complete. You can then comment out this function being called. 
    """
    plt.figure(figsize=(12, 8))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(frames_dir, frame_names[ann_frame_idx])))
    show_chosen_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    plt.show()

    print("Press key to continue")
    input()
    print("Continuing")

def save_processed_images(frames_dir, frame_names, video_segments, frame_stride=1, output_dir="sam2_outputs"):

    print("Saving processed images")
    plt.close("all")

    fig = plt.figure(figsize=(6, 4))
    for out_frame_idx in range(0, len(frame_names), frame_stride):
        image_path = os.path.join(frames_dir, frame_names[out_frame_idx])
        image = Image.open(image_path)
        image_size = image.size  # Ensure masks match this size

        plt.title(f"frame {out_frame_idx}")
        plt.imshow(image, animated=True)

        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id, image_size=image_size)

        output_path = os.path.join(BASE_PATH, output_dir, SAMPLE_FRAMES_DIR)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Crude way of handling clashing filename so I don't overwrite outputs from different test runs 
        output_path = os.path.join(output_path, f'frame{out_frame_idx}.png')
        counter = 1
        while os.path.exists(output_path):
            # add counter before filename so that each run appears separately when browsing in the folder
            output_path = os.path.join(output_path, f'{counter}_frame{out_frame_idx}.png')  
            counter += 1

        # print("Saving image", out_frame_idx)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.clf()  # clear the figure to avoid overlapping of images

def overlay_box(box, ax):   
    """
    Expects bbox in x_min,y_min,x_max,y_max format, and converts to x,y,w,h for plt.Rectangle
    """
    x1, y1 = box[0]
    x2, y2 = box[1]
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)

parser = argparse.ArgumentParser(description="Testing Offloading Options for SAM2")
parser.add_argument("--visualize", action="store_true", help="Visualize bbox of first frame before running the model")
parser.add_argument("--offload_video", action="store_true", help="Offload video to CPU")
parser.add_argument("--offload_state", action="store_true", help="Offload state to CPU")
parser.add_argument("--frames_dir", type=str, default="sample_frames", help="Directory containing the frames to input to the model")
args = parser.parse_args()

SAMPLE_FRAMES_DIR = args.frames_dir
visualize_bbox = args.visualize

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

# Change to absolute path
OUTPUT_DIR = os.path.join(BASE_PATH, OUTPUT_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

frames_dir = os.path.join(BASE_PATH, SAMPLE_FRAMES_DIR)
frame_names = [
    p for p in os.listdir(frames_dir)
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
# display_initial_frame(frames_dir, frame_names)

inference_state = predictor.init_state(video_path=frames_dir, offload_video_to_cpu=args.offload_video, offload_state_to_cpu=args.offload_state)
predictor.reset_state(inference_state)

ann_frame_idx = 0   
ann_obj_id = 1      # this is the ID for the player, we could have chosen any integer

# No longer using points as prompt since we cannot extract that automatically 
# points = np.array([[35,40], [20,60]], dtype = np.float32) 
box = []
if args.frames_dir == "sample_frames":
    box = np.array([[8,7], [50,131]])
elif args.frames_dir == "sample_frames2":
    box = np.array([[3,3], [50,125]])       
labels = np.array([1,1], np.int32)      # this tells the predictor that the points in the previous line correspond to the same target object

# Display the initial frame with the bounding box overlay
if visualize_bbox:
    plt.figure(figsize=(12, 8))
    plt.title(f"Frame {ann_frame_idx}")
    image = Image.open(os.path.join(frames_dir, frame_names[ann_frame_idx]))
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
# display_annotation_frame(frames_dir, frame_names, ann_frame_idx, points, labels, out_mask_logits, out_obj_ids)

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

save_processed_images(frames_dir, frame_names, video_segments, frame_stride=1, output_dir=OUTPUT_DIR)

"""
TODO:
    1) Fix the issue of the complete tracklet not fitting into VRAM; in that case split the video into two, and input the mask of the last frame of the first half into the second half
    2) use predictor.add_new_points_or_box(), and see if the results using a bounding box as prompt are as good as using 2 points as a prompt (since bounding boxes can be automatically extracted from a lightweight model like YOLO). 
    3) if the output masks are good, then see if YOLOv11 generates good bounding boxes for a few sample tracklets

For testing bbox: here is a manually annotated bbox for frame 0, to pass into SAM2. This will simulate getting an automatic bounding box with the first player
    "x": "27.55",
    "y": "69.90",
    "width": "42.04",
    "height": "127.14"

    4) Figure out what format sam2 expects bounding boxes to be in
"""

"""
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
"""