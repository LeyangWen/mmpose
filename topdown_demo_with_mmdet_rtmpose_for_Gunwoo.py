# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import time
import shlex
import tempfile
from pathlib import Path
import shutil
import subprocess
from argparse import ArgumentParser
import glob

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
from utils_gunwoo import load_json_arr, filter_subject_using_center_of_joints_with_disqualify

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def disassemble_video(video_path, output_dir, fps):
    """Generate frames from video file."""
    dest = os.path.join(output_dir, "img%05d.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    subprocess.run(shlex.split("/home/ubuntu/anaconda3/envs/openmmlab/bin/ffmpeg -i \"{}\" -vsync 1 -q:v 1 -r {} -an -y \"{}\"".format(
        video_path, fps, dest)), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(glob.glob(os.path.join(output_dir, "*")))

def downsample_video(frame_list):
    """Downsample the video to every 5th frame and rename the frames."""
    # Step 1: Select every 5th frame starting with the 0th frame
    selected_frames = frame_list[::5]

    # Step 2: Remove frames not in the selected list
    # Create a set of selected frames for faster lookup
    selected_frames_set = set(selected_frames)

    # Iterate over the original frame list
    for frame in frame_list:
        if frame not in selected_frames_set:
            try:
                os.remove(frame)  # Delete the file from disk
                print(f"Deleted: {frame}")
            except OSError as e:
                print(f"Error deleting {frame}: {e}")

    # Step 3: Rename the selected frames in order
    destination_folder = os.path.dirname(selected_frames[0])
    new_frame_list = []

    for idx, frame in enumerate(selected_frames):
        new_index = idx + 1  # Start counting from 1
        new_filename = f"img{new_index:05d}.jpg"  # Format: img00001.jpg
        new_frame = os.path.join(destination_folder, new_filename)
        os.rename(frame, new_frame)
        new_frame_list.append(new_frame)

    return new_frame_list


def crop_and_save_hands(all_keyps_rtm, frame_list, frames_folder, video_name,
                        hand_padding=0.30, hand_visibility_threshold=5.0):
    """
    Crop individual hands (left and right) from frames using filtered subject keypoints.

    Uses elbow + wrist + hand keypoints for bbox calculation.
    Checks hand keypoints only for visibility (strict: ANY keypoint < threshold = skip).

    Args:
        all_keyps_rtm: Filtered subject keypoints, shape (num_frames, 46, 3)
        frame_list: List of frame file paths
        frames_folder: Base folder containing frames
        video_name: Video filename without extension
        hand_padding: Padding percentage for hand crops (default: 0.30 for 30%)
        hand_visibility_threshold: Skip crop if ANY hand keypoint < threshold (default: 5.0)

    Returns:
        dict with cropping statistics
    """
    import cv2
    import numpy as np
    import os

    print(f"\n{'=' * 60}")
    print(f"CROPPING INDIVIDUAL HANDS")
    print(f"{'=' * 60}")
    print(f"Configuration:")
    print(f"  Hand padding: {hand_padding * 100}%")
    print(f"  Visibility threshold: {hand_visibility_threshold}")
    print(f"  Visibility check: Skip if ANY hand keypoint < threshold")
    print(f"  BBox includes: Elbow + Wrist + Hand")

    # Create output directories
    left_hand_dir = os.path.join(frames_folder, f"{video_name}_left_hand")
    right_hand_dir = os.path.join(frames_folder, f"{video_name}_right_hand")
    os.makedirs(left_hand_dir, exist_ok=True)
    os.makedirs(right_hand_dir, exist_ok=True)

    print(f"\nOutput folders:")
    print(f"  Left hand: {left_hand_dir}/")
    print(f"  Right hand: {right_hand_dir}/")

    # Statistics
    num_frames = len(frame_list)
    left_cropped = 0
    left_skipped = 0
    right_cropped = 0
    right_skipped = 0

    print(f"\nProcessing {num_frames} frames...")

    for frame_idx in range(num_frames):
        frame_path = frame_list[frame_idx]
        frame_name = os.path.basename(frame_path)

        # Read frame once
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}, skipping...")
            left_skipped += 1
            right_skipped += 1
            continue

        img_height, img_width = frame.shape[:2]

        # ================================================================
        # LEFT HAND
        # ================================================================
        # Check visibility (hand keypoints only: 2-22)
        left_hand_scores = all_keyps_rtm[frame_idx, 2:23, 2]  # 21 hand keypoints

        if np.any(left_hand_scores < hand_visibility_threshold):
            # Skip - at least one keypoint has low visibility
            left_skipped += 1
        else:
            # Get keypoints for bbox (elbow + wrist + hand: 0-22)
            left_keypoints = all_keyps_rtm[frame_idx, 0:23, :2]  # (23, 2) - x,y only

            # Calculate bounding box from keypoints
            x_min = np.min(left_keypoints[:, 0])
            y_min = np.min(left_keypoints[:, 1])
            x_max = np.max(left_keypoints[:, 0])
            y_max = np.max(left_keypoints[:, 1])

            # Add padding
            width = x_max - x_min
            height = y_max - y_min
            pad_x = width * hand_padding / 2
            pad_y = height * hand_padding / 2

            x_min -= pad_x
            x_max += pad_x
            y_min -= pad_y
            y_max += pad_y

            # Clip to image boundaries
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(img_width, int(x_max))
            y_max = min(img_height, int(y_max))

            # Crop and save
            cropped_left = frame[y_min:y_max, x_min:x_max]

            # Check if crop is valid (non-zero size)
            if cropped_left.shape[0] > 0 and cropped_left.shape[1] > 0:
                output_path = os.path.join(left_hand_dir, frame_name)
                cv2.imwrite(output_path, cropped_left)
                left_cropped += 1
            else:
                output_path = os.path.join(left_hand_dir, frame_name)
                cv2.imwrite(output_path, frame)
                left_skipped += 1

        # ================================================================
        # RIGHT HAND
        # ================================================================
        # Check visibility (hand keypoints only: 25-45)
        right_hand_scores = all_keyps_rtm[frame_idx, 25:46, 2]  # 21 hand keypoints

        if np.any(right_hand_scores < hand_visibility_threshold):
            # Skip - at least one keypoint has low visibility
            right_skipped += 1
        else:
            # Get keypoints for bbox (elbow + wrist + hand: 23-45)
            right_keypoints = all_keyps_rtm[frame_idx, 23:46, :2]  # (23, 2) - x,y only

            # Calculate bounding box from keypoints
            x_min = np.min(right_keypoints[:, 0])
            y_min = np.min(right_keypoints[:, 1])
            x_max = np.max(right_keypoints[:, 0])
            y_max = np.max(right_keypoints[:, 1])

            # Add padding
            width = x_max - x_min
            height = y_max - y_min
            pad_x = width * hand_padding / 2
            pad_y = height * hand_padding / 2

            x_min -= pad_x
            x_max += pad_x
            y_min -= pad_y
            y_max += pad_y

            # Clip to image boundaries
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(img_width, int(x_max))
            y_max = min(img_height, int(y_max))

            # Crop and save
            cropped_right = frame[y_min:y_max, x_min:x_max]

            # Check if crop is valid (non-zero size)
            if cropped_right.shape[0] > 0 and cropped_right.shape[1] > 0:
                output_path = os.path.join(right_hand_dir, frame_name)
                cv2.imwrite(output_path, cropped_right)
                right_cropped += 1
            else:
                output_path = os.path.join(right_hand_dir, frame_name)
                cv2.imwrite(output_path, frame)
                right_skipped += 1

        # Progress indicator
        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames...")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"HAND CROPPING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Left hand:")
    print(f"  Cropped: {left_cropped} frames")
    print(f"  Skipped (low visibility): {left_skipped} frames")
    print(f"  Success rate: {left_cropped / num_frames * 100:.1f}%")
    print(f"\nRight hand:")
    print(f"  Cropped: {right_cropped} frames")
    print(f"  Skipped (low visibility): {right_skipped} frames")
    print(f"  Success rate: {right_cropped / num_frames * 100:.1f}%")
    print(f"{'=' * 60}\n")

    return {
        'left_cropped': left_cropped,
        'left_skipped': left_skipped,
        'right_cropped': right_cropped,
        'right_skipped': right_skipped,
        'total_frames': num_frames
    }

def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def get_video_resolution(video_path):
    """Get the width and height of the video using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if we got a valid response
    if result.stdout.strip() == "":
        # Print error message if `ffprobe` fails to retrieve the resolution
        print(f"Error retrieving resolution for {video_path}: {result.stderr.strip()}")
        raise ValueError("Failed to retrieve video resolution. Check if the file path is correct and if the video is accessible.")

    # Split the output and convert to integers
    try:
        width, height = map(int, result.stdout.strip().split('x'))
        print(f"Width: {width}, Height: {height}")
    except ValueError:
        raise ValueError("Unexpected output format from ffprobe. Could not parse width and height.")

    return width, height


# def get_video_resolution(video_path):
#     """Get the width and height of the video using ffmpeg."""
#     cmd = [
#         "/home/ubuntu/anaconda3/envs/mediapipe/bin/ffmpeg",
#         "-i", video_path,
#         "-v", "error",
#         "-select_streams", "v:0",
#         "-show_entries", "stream=width,height",
#         "-of", "csv=s=x:p=0"
#     ]
#     result = subprocess.run(cmd, capture_output=True, text=True)
#     width, height = map(int, result.stdout.strip().split('x'))
#     return width, height

def is_portrait(width, height):
    """Check if the video is in portrait orientation."""
    return height > width

def needs_downscaling(width, height):
    """Check if the video resolution is greater than 1080 FHD (greater than 1080 for width or greater than 1920 for height (portrait only))."""
    return (width > 1080 or height > 1920)

def calculate_new_dimensions(width, height):
    """Calculate new dimensions to fit within 1080x1920 while keeping the aspect ratio and ensuring even dimensions."""
    aspect_ratio = width / height
    # Scale down based on the largest dimension for portrait videos
    if height > 1920:
        new_height = 1920
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = 1080
        new_height = int(new_width / aspect_ratio)

    # Ensure both dimensions are even
    new_width = new_width if new_width % 2 == 0 else new_width - 1
    new_height = new_height if new_height % 2 == 0 else new_height - 1

    print(f"New width: {new_width}, New height: {new_height}")

    return new_width, new_height

def downscale_to_fhd(video_path, output_path, new_width, new_height):
    """Downscale a 4K video to Full HD (1080p) using ffmpeg."""

    scale_option = f"scale={new_width}:{new_height}"

    cmd = [
        "/home/ubuntu/anaconda3/envs/mediapipe/bin/ffmpeg",
        "-i", video_path,
        "-vf", scale_option,  # scale to 1080p while keeping the aspect ratio
        "-c:a", "copy",          # copy audio stream without re-encoding
        output_path
    ]
    subprocess.run(cmd, check=True,stdout=subprocess.DEVNULL)


# def split_instances_with_visibility(pred_instances):
#     """Custom split that preserves visibility scores if they exist."""
#     if pred_instances is None:
#         return []
#
#     results = []
#     num_instances = len(pred_instances.keypoints)
#
#     for i in range(num_instances):
#         instance_dict = {
#             'keypoints': pred_instances.keypoints[i].tolist(),
#             'keypoint_scores': pred_instances.keypoint_scores[i].tolist()
#         }
#
#         # Add visibility scores if they exist
#         if hasattr(pred_instances, 'keypoints_visible'):
#             instance_dict['keypoints_visible'] = pred_instances.keypoints_visible[i].tolist()
#             print(
#                 f"Found visibility scores! Range: {min(instance_dict['keypoints_visible']):.2f} - {max(instance_dict['keypoints_visible']):.2f}")
#         else:
#             print("No keypoints_visible attribute found")
#
#         # Add bbox if available
#         if hasattr(pred_instances, 'bboxes'):
#             instance_dict['bbox'] = pred_instances.bboxes[i].tolist()
#
#         results.append(instance_dict)
#
#     return results

def split_instances_with_visibility(instances):  # Removed ": InstanceData" and "-> List[InstanceData]"
    """Convert instances into a list where each element is a dict that contains
    information about one instance, INCLUDING visibility scores if available."""
    results = []

    # return an empty list if there is no instance detected by the model
    if instances is None:
        return results

    for i in range(len(instances.keypoints)):
        result = dict(
            keypoints=instances.keypoints[i].tolist(),
            keypoint_scores=instances.keypoint_scores[i].tolist(),
        )

        # Add visibility scores if they exist
        if hasattr(instances, 'keypoints_visible'):
            result['keypoints_visible'] = instances.keypoints_visible[i].tolist()

        # Handle bbox the same way as original
        if 'bboxes' in instances:
            result['bbox'] = instances.bboxes[i].tolist(),  # Note the comma - keeping original structure
            if 'bbox_scores' in instances:
                result['bbox_score'] = instances.bbox_scores[i]

        results.append(result)

    return results

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument(
        '--frames_folder',
        type=str,
        default='',
        help='Folder to save the disassembled frames of the video.')
    parser.add_argument(
        '--fps',
        type=int,
        default=15,
        help='FPS for disassembling the video')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if input_type == 'image':

        # inference
        pred_instances = process_one_image(args, args.input, detector,
                                           pose_estimator, visualizer)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type in ['webcam', 'video']:

        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            input_filename = os.path.basename(args.input).split('.')[0]
            frames_dir = os.path.join(args.frames_folder, input_filename)
            # width, height = get_video_resolution(args.input)

            # Check if the video is portrait and 4K
            # if is_portrait(width, height) and needs_downscaling(width, height):
            #     print("The video is portrait and 4K, resizing to Full HD (1080p).")
            #
            #     # Calculate the new dimensions to fit within FHD while keeping aspect ratio
            #     new_width, new_height = calculate_new_dimensions(width, height)
            #
            # # Create a temporary directory for the FHD video
            #     with tempfile.TemporaryDirectory() as temp_dir:
            #         original_filename = Path(args.input).name
            #         temp_output_path = str(Path(temp_dir) / original_filename)
            #         # temp_output_path = Path(temp_dir) / "fhd_video.mp4"
            #
            #         # Downscale the video to FHD
            #         downscale_to_fhd(args.input, temp_output_path,new_width, new_height)
            #         print(f"FHD video saved temporarily at {temp_output_path}")
            #         frame_list = disassemble_video(temp_output_path, frames_dir,fps=args.fps)
            #         #frame_list = downsample_video(frame_list)
            # else:
            frame_list = disassemble_video(args.input, frames_dir, fps=args.fps)
                #frame_list = downsample_video(frame_list)
        #     cap = cv2.VideoCapture(args.input)
        # captured_fps = cap.get(cv2.CAP_PROP_FPS)
        # print(f'captured fps is {captured_fps}')
        #cap.set(cv2.cv.CAP_PROP_FPS, 15)
        #captured_fps = cap.get(cv2.CAP_PROP_FPS)
        #print(f'captured fps after setting is {captured_fps}')
        video_writer = None
        pred_instances_list = []
        frame_idx = 0
        # logging_info_folder = os.path.join(args.output_root, 'logging_info',os.path.splitext(os.path.basename(args.input))[0])
        # os.makedirs(logging_info_folder, exist_ok=True)
        for frame_loc in frame_list:
            frame = cv2.imread(frame_loc)
            frame_idx += 1

            # # Save the frame as a numpy in the logging folder
            # frame_save_loc = os.path.join(logging_info_folder, f'frame_{frame_idx}.npy')
            # np.save(frame_save_loc, frame)


            # topdown pose estimation
            pred_instances = process_one_image(args, frame, detector,
                                               pose_estimator, visualizer,
                                               0.001)

            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances_with_visibility(pred_instances)))

            # output videos
            # output videos
            if output_file:
                frame_vis = visualizer.get_image()

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        args.fps,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

            if args.show:
                # press ESC to exit
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                time.sleep(args.show_interval)

        if video_writer:
            video_writer.release()

        # cap.release()

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        with open(args.pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')

        # Subject tracking using elbow and hand kps.
        all_bboxes, all_keyps_bef = load_json_arr(args.pred_save_path)
        # all_keyps_rtm = filter_subject_using_center_of_joints_with_disqualify(all_keyps_bef, window=4, vis_threshold=5,
        #                                                                    num_joints=46)

        all_keyps_rtm, selected_indices = filter_subject_using_center_of_joints_with_disqualify(all_keyps_bef, window=4,
                                                                                                vis_threshold=5,
                                                                                                num_joints=46)
        final_filtered_keyps = np.copy(all_keyps_rtm)
        subject_filtered_kps_path = f'{args.output_root}/{input_filename}_subject_filtered_kps.npy'
        np.save(subject_filtered_kps_path, final_filtered_keyps)



        # Crop and save images using filtered subject's bbox
        print(f"Cropping images with {args.bbox_padding * 100}% padding...")
        frames_dir = os.path.join(args.frames_folder, input_filename)
        cropped_dir = os.path.join(args.frames_folder, f"{input_filename}_cropped")
        os.makedirs(cropped_dir, exist_ok=True)

        num_frames = len(selected_indices)
        cropped_count = 0
        copied_count = 0

        for frame_idx in range(num_frames):
            frame_path = frame_list[frame_idx]
            output_path = os.path.join(cropped_dir, os.path.basename(frame_path))

            # Check if valid instance was selected
            instance_idx = selected_indices[frame_idx]

            if instance_idx >= 0 and len(all_bboxes[frame_idx]) > 0:
                # Get bbox from filtered subject
                bbox = all_bboxes[frame_idx][instance_idx]  # [x1, y1, x2, y2, score]

                # Read frame
                frame = cv2.imread(frame_path)
                if frame is None:
                    print(f"Warning: Could not read frame {frame_path}, skipping...")
                    continue

                img_height, img_width = frame.shape[:2]

                # Add padding
                x1, y1, x2, y2 = bbox[:4]
                width = x2 - x1
                height = y2 - y1
                pad_x = width * args.bbox_padding / 2
                pad_y = height * args.bbox_padding / 2

                x1 -= pad_x
                x2 += pad_x
                y1 -= pad_y
                y2 += pad_y

                # Clip to image boundaries
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(img_width, int(x2))
                y2 = min(img_height, int(y2))

                # Crop and save
                cropped = frame[y1:y2, x1:x2]
                cv2.imwrite(output_path, cropped)
                cropped_count += 1
            else:
                # No valid subject detected - copy original frame
                frame = cv2.imread(frame_path)
                if frame is not None:
                    cv2.imwrite(output_path, frame)
                    copied_count += 1

        print(f'Image cropping complete:')
        print(f'  Cropped: {cropped_count} frames')
        print(f'  Copied original: {copied_count} frames')
        print(f'  Saved to: {cropped_dir}/')

        # Crop individual hands (elbow + wrist + hand)
        hand_crop_stats = crop_and_save_hands(
            all_keyps_rtm=all_keyps_rtm,
            frame_list=frame_list,
            frames_folder=args.frames_folder,
            video_name=input_filename,
            hand_padding=args.hand_padding,
            hand_visibility_threshold=args.hand_visibility_threshold
        )

        print("Processing hand poses for grip classification...")


if __name__ == '__main__':
    main()
