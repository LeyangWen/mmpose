import json_tricks as json
import numpy as np


def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        all_bboxes, all_keyps = [], []
        for frame_idx in range(len(json_data['instance_info'])):
            bboxes = []
            keyps = []
            for instance in range(len(json_data['instance_info'][frame_idx]['instances'])):
                bbox_value = json_data['instance_info'][frame_idx]['instances'][instance]['bbox'][0]
                bbox_value.append(json_data['instance_info'][frame_idx]['instances'][instance]['bbox_score'])
                bboxes.append(bbox_value)
                keyp_values = json_data['instance_info'][frame_idx]['instances'][instance]['keypoints']
                keyp_scores = json_data['instance_info'][frame_idx]['instances'][instance]['keypoint_scores']
                [keyp_values[i].append(keyp_scores[i]) for i in range(len(keyp_values))]
                keyps.append(keyp_values)
            all_bboxes.append(np.array(bboxes))
            all_keyps.append(np.array(keyps))
    print('done')
    return all_bboxes, all_keyps


def get_rtm_keyps_37(all_keyps, num_joints=37):
    """
    Extract 37 keypoints matching YOUR EXACT pattern
    Takes: (num_instances, 133, 3)
    Returns: (num_instances, 37, 3)
    """
    rtm_keyps = np.zeros((len(all_keyps), num_joints, 3))

    rtm_keyps[:, :17, :3] = all_keyps[:, :17, :3]
    rtm_keyps[:, 17, :3] = all_keyps[:, 96, :3]
    rtm_keyps[:, 18, :3] = all_keyps[:, 100, :3]
    rtm_keyps[:, 19, :3] = all_keyps[:, 108, :3]
    rtm_keyps[:, 20, :3] = all_keyps[:, 117, :3]
    rtm_keyps[:, 21, :3] = all_keyps[:, 121, :3]
    rtm_keyps[:, 22, :3] = all_keyps[:, 129, :3]
    rtm_keyps[:, 23, :3] = (rtm_keyps[:, 11, :3] + rtm_keyps[:, 12, :3]) / 2  # Pelvis
    rtm_keyps[:, 24, :3] = (rtm_keyps[:, 5, :3] + rtm_keyps[:, 6, :3]) / 2  # Shoulder
    rtm_keyps[:, 25, :3] = (rtm_keyps[:, 3, :3] + rtm_keyps[:, 4, :3]) / 2  # Head
    return rtm_keyps


def filter_subject_using_center_of_joints_with_disqualify(all_keyps_bef, window=4, vis_threshold=5, num_joints=46):
    """
    Filter to select one subject per frame based on tracking.

    MODIFICATION: Now also returns selected instance indices for each frame.

    Returns:
        all_keyps: numpy array of shape (num_frames, num_joints, 3) - filtered keypoints
        selected_indices: numpy array of shape (num_frames,) - which instance was selected per frame
                         -1 indicates no valid instance found
    """

    def calculate_subject_size(keypoints):
        """Calculate bounding box size of the subject"""
        valid_points = keypoints[keypoints[:, 2] > vis_threshold]  # Only consider points with good confidence
        if len(valid_points) < 2:
            return 0
        x_min, x_max = np.min(valid_points[:, 0]), np.max(valid_points[:, 0])
        y_min, y_max = np.min(valid_points[:, 1]), np.max(valid_points[:, 1])
        return (x_max - x_min) * (y_max - y_min)

    all_keyps = []
    selected_indices = []  # NEW: Track which instance was selected
    prev_frames = [0]

    for frame_idx, frame in enumerate(all_keyps_bef):
        if len(frame) > 0:
            frame_keyps = [[0, 0, 0]] * num_joints
            selected_idx = -1  # Default to -1 (no selection)

            # Special handling for first frame or when previous frames are empty
            if (frame_idx == 0) or not np.array(prev_frames).any():
                # Store metrics for all subjects
                subjects = []
                confidences = []

                # Calculate confidence and size for each subject
                for instance_idx, instance in enumerate(frame):
                    instance_expanded = np.expand_dims(instance, axis=0)
                    rtm_keyps = get_rtm_keyps_37(instance_expanded, num_joints)
                    avg_conf = np.mean(rtm_keyps[0, :, 2])
                    size = calculate_subject_size(rtm_keyps[0])

                    subjects.append({
                        'keypoints': rtm_keyps[0],
                        'confidence': avg_conf,
                        'size': size,
                        'instance_idx': instance_idx  # NEW: Track instance index
                    })
                    confidences.append(avg_conf)

                # Calculate mean confidence threshold
                mean_confidence = np.mean(confidences)

                # Filter subjects above mean confidence and find the largest one
                qualified_subjects = [s for s in subjects if s['confidence'] >= mean_confidence]

                if qualified_subjects:
                    # Select subject with largest size among qualified subjects
                    selected_subject = max(qualified_subjects, key=lambda x: x['size'])
                    frame_keyps = selected_subject['keypoints']
                    selected_idx = selected_subject['instance_idx']  # NEW: Store selected index

                if frame_idx == 0:
                    prev_frames[0] = frame_keyps
                else:
                    prev_frames.append(frame_keyps)
                    if len(prev_frames) > window:
                        prev_frames.pop(0)

            else:
                # Original logic for subsequent frames
                distance = []
                instance_dict = {}
                for instance_idx, instance in enumerate(frame):
                    instance_expanded = np.expand_dims(instance, axis=0)
                    rtm_keyps = get_rtm_keyps_37(instance_expanded, num_joints)
                    instance_dict[instance_idx] = rtm_keyps[0]
                    frames_matrix = np.concatenate((np.array(prev_frames), rtm_keyps), axis=0)

                    average_x_previous = np.average(frames_matrix[:-1, :, 0])
                    average_y_previous = np.average(frames_matrix[:-1, :, 1])
                    average_x_subject = np.average(frames_matrix[-1, :, 0])
                    average_y_subject = np.average(frames_matrix[-1, :, 1])
                    distance.append(np.linalg.norm(np.array([average_x_previous, average_y_previous]) -
                                                   np.array([average_x_subject, average_y_subject])))

                min_index = distance.index(min(distance))
                frame_keyps = instance_dict[min_index]
                selected_idx = min_index  # NEW: Store selected index
                prev_frames.append(frame_keyps)
                if len(prev_frames) > window:
                    prev_frames.pop(0)

            all_keyps.append(frame_keyps)
            selected_indices.append(selected_idx)  # NEW: Append selected index
        else:
            if isinstance(prev_frames[0], int):
                prev_frames[0] = [[0, 0, 0]] * num_joints
            else:
                prev_frames.append([[0, 0, 0]] * num_joints)
            all_keyps.append([[0, 0, 0]] * num_joints)
            selected_indices.append(-1)  # NEW: No valid instance

    return np.array(all_keyps), np.array(selected_indices)


def extrapolate_point(frames, index, part_index, filtered_keyps):
    if index == 0 or index == len(frames) - 1:
        for dims in range(frames.shape[2]):
            filtered_keyps[index, part_index, dims] = frames[index, part_index, dims]
        return filtered_keyps
    if (np.any(frames[index - 1])) and (np.any(frames[index + 1])):
        for dims in range(frames.shape[2]):
            prev = frames[index - 1, part_index, dims]
            next = frames[index + 1, part_index, dims]
            if frames.shape[2] == 3:
                if dims == frames.shape[2] - 1:
                    filtered_keyps[index, part_index, dims] = frames[index, part_index, dims]
                else:
                    filtered_keyps[index, part_index, dims] = (prev + next) // 2
            if frames.shape[2] == 4:
                if dims == frames.shape[2] - 1:
                    filtered_keyps[index, part_index, dims] = frames[index, part_index, dims]
                else:
                    filtered_keyps[index, part_index, dims] = (prev + next) / 2
    return filtered_keyps


def smooth_point(frames, index, part_index, window=8, vis_threshold=5.0):
    """Smooth points out based on their adjacent points"""
    start_index = max(0, index - window // 2)
    end_index = min(len(frames), index + window // 2 + 1)
    confidence_score = frames[index, part_index, -1]
    dims_sum = []
    for dims in range(frames.shape[2]):
        if dims != frames.shape[2] - 1:
            dims_sum.append(0.0)
    weight = 0.0

    for i in range(start_index, end_index):
        point = frames[i, part_index]
        if np.any(frames[i]):
            # Weight each point by its distance from the target point
            multiplier = (abs(i - index) - window) ** 2 / (window ** 2)
            for dims in range(frames.shape[2]):
                if dims != frames.shape[2] - 1:
                    dims_sum[dims] += point[dims] * multiplier
            weight += multiplier

    if frames.shape[2] == 3:
        if np.any(frames[start_index:end_index, part_index, 2] < vis_threshold):
            confidence_score = 0.0

    # if np.count_nonzero(frames[start_index:end_index,part_index,-1] < 0.3) > 3:
    #     confidence_score = 0.0

    if weight == 0.0:
        if frames.shape[2] == 3:
            return frames[index, part_index, 0], frames[index, part_index, 1], confidence_score
        if frames.shape[2] == 4:
            return frames[index, part_index, 0], frames[index, part_index, 1], frames[
                index, part_index, 2], confidence_score

    if frames.shape[2] == 3:
        return int(dims_sum[0] / weight), int(dims_sum[1] / weight), confidence_score
    if frames.shape[2] == 4:
        return dims_sum[0] / weight, dims_sum[1] / weight, dims_sum[2] / weight, confidence_score


def extrapolate_and_smooth(all_keyps, vis_threshold=5.0):
    filtered_keyps = np.zeros(all_keyps.shape)
    final_filtered_keyps = np.zeros(all_keyps.shape)
    for i, frame in enumerate(all_keyps):
        if not np.any(frame):  # Check if the frame is empty and has no prediction
            continue
        for j, part in enumerate(frame):
            filtered_keyps = extrapolate_point(all_keyps, i, j, filtered_keyps)
    for i, frame in enumerate(all_keyps):
        if not np.any(frame):  # Check if the frame is empty and has no prediction
            continue
        for j, part in enumerate(frame):
            window = 8  # Default window size
            # if j in [ref.rtm_pose_keypoints_vicon_dataset.index('left_pinky'),ref.rtm_pose_keypoints_vicon_dataset.index('right_pinky'),ref.rtm_pose_keypoints_vicon_dataset.index('left_index'),ref.rtm_pose_keypoints_vicon_dataset.index('right_index'),ref.rtm_pose_keypoints_vicon_dataset.index('left_middle_mcp'),ref.rtm_pose_keypoints_vicon_dataset.index('right_middle_mcp')]:
            #     window = 4 # Smaller window size for hand keypoints
            if all_keyps.shape[2] == 3:
                final_filtered_keyps[i, j, 0], final_filtered_keyps[i, j, 1], final_filtered_keyps[
                    i, j, 2] = smooth_point(filtered_keyps, i, j, window=window, vis_threshold=vis_threshold)
            if all_keyps.shape[2] == 4:
                final_filtered_keyps[i, j, 0], final_filtered_keyps[i, j, 1], final_filtered_keyps[i, j, 2], \
                final_filtered_keyps[i, j, 3] = smooth_point(filtered_keyps, i, j, window=window,
                                                             vis_threshold=vis_threshold)
    return final_filtered_keyps


def calculate_wrist_angles_conservative(smoothed_keyps):
    """
    Calculate wrist angles with conservative PM for RSI assessment
    ALWAYS assumes flexion (worst case) for safety

    Args:
        smoothed_keyps: Smoothed keypoints array, shape (num_frames, 46, 3)

    Returns:
        left_wrist_data: Dict with 'angle' (degrees) and 'PM' (posture multiplier)
        right_wrist_data: Dict with 'angle' (degrees) and 'PM' (posture multiplier)
    """
    num_frames = smoothed_keyps.shape[0]

    left_wrist_data = {
        'angle': np.zeros(num_frames),
        'PM': np.ones(num_frames)  # Default to 1.0 (neutral)
    }

    right_wrist_data = {
        'angle': np.zeros(num_frames),
        'PM': np.ones(num_frames)
    }

    for frame_idx in range(num_frames):
        # LEFT HAND
        left_elbow = smoothed_keyps[frame_idx, 0, :2]
        left_wrist = smoothed_keyps[frame_idx, 2, :2]  # Hand wrist
        left_middle_base = smoothed_keyps[frame_idx, 11, :2]  # Index 2+9

        if np.linalg.norm(left_wrist - left_elbow) > 0 and np.linalg.norm(left_middle_base - left_wrist) > 0:
            # Calculate angle deviation
            forearm_vec = left_wrist - left_elbow
            hand_vec = left_middle_base - left_wrist

            cos_angle = np.dot(forearm_vec, hand_vec) / (
                    np.linalg.norm(forearm_vec) * np.linalg.norm(hand_vec)
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))
            deviation = 180.0 - angle_deg

            # Calculate PM using flexion formula (conservative)
            if deviation < 5:
                PM = 1.0  # Neutral
            else:
                # Flexion formula: PM = 1.2 · e^(0.009·P) - 0.2
                PM = 1.2 * np.exp(0.009 * deviation) - 0.2

            left_wrist_data['angle'][frame_idx] = deviation
            left_wrist_data['PM'][frame_idx] = PM
        # else: keeps default values (angle=0, PM=1.0)

        # RIGHT HAND
        right_elbow = smoothed_keyps[frame_idx, 23, :2]
        right_wrist = smoothed_keyps[frame_idx, 25, :2]  # Hand wrist
        right_middle_base = smoothed_keyps[frame_idx, 34, :2]  # Index 25+9

        if np.linalg.norm(right_wrist - right_elbow) > 0 and np.linalg.norm(right_middle_base - right_wrist) > 0:
            # Calculate angle deviation
            forearm_vec = right_wrist - right_elbow
            hand_vec = right_middle_base - right_wrist

            cos_angle = np.dot(forearm_vec, hand_vec) / (
                    np.linalg.norm(forearm_vec) * np.linalg.norm(hand_vec)
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))
            deviation = 180.0 - angle_deg

            # Calculate PM using flexion formula (conservative)
            if deviation < 5:
                PM = 1.0
            else:
                PM = 1.2 * np.exp(0.009 * deviation) - 0.2

            right_wrist_data['angle'][frame_idx] = deviation
            right_wrist_data['PM'][frame_idx] = PM

    return left_wrist_data, right_wrist_data
