#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
from collections import defaultdict
from fvcore.common.file_io import PathManager
from datasets.ava_eval_helper import read_exclusions

logger = logging.getLogger(__name__)
FPS = 30
AVA_VALID_FRAMES = range(902, 1799)


def load_image_lists(cfg, is_train):
    """
    Loading image paths from corresponding files.

    Args:
        cfg (CfgNode): config.
        is_train (bool): if it is training dataset or not.

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """
    # frame_list_dir is /data3/ava/frame_lists/
    # contains 'train.csv' and 'val.csv'
    print("@@@@@@@@@@ ava_helper. load_image_lists-->argus: is_train: ", is_train)
    print("##### cfg.AVA.FRAME_LIST_DIR: ", cfg.AVA.FRAME_LIST_DIR) # where?
    print("##### cfg.AVA.TRAIN_LISTS: ", cfg.AVA.TRAIN_LISTS)

    list_filenames = [
        os.path.join(cfg.AVA.FRAME_LIST_DIR, filename)
        for filename in (
            cfg.AVA.TRAIN_LISTS if is_train else cfg.AVA.TEST_LISTS
        )
    ]
    print("##### list_filenames: ", list_filenames) # 'datasets/AVA/frame_lists/train.csv'

    image_paths = defaultdict(list)
    video_name_to_idx = {}
    video_idx_to_name = []
    for list_filename in list_filenames: # list_filename: 'datasets/AVA/frame_lists/train.csv'
        with PathManager.open(list_filename, "r") as f:
            f.readline()
            for line in f:
                row = line.split()
                # The format of each row should follow:
                # original_vido_id video_id frame_id path labels.
                # print("******************************* row: ", row)
                # print("******************************* row: ", len(row))
                assert len(row) == 5
                video_name = row[0]

                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name.append(video_name)

                data_key = video_name_to_idx[video_name]

                image_paths[data_key].append(
                    os.path.join(cfg.AVA.FRAME_DIR, row[3])
                )

    image_paths = [image_paths[i] for i in range(len(image_paths))]
    # print("image_paths: ", image_paths) # e.g., datasets/AVA/frames/SHBMiL5f_3Q/SHBMiL5f_3Q_005261.jpg

    logger.info(
        "Finished loading image paths from: %s" % ", ".join(list_filenames)
    )
    print("##### image_paths lengh: ", len(image_paths))
    print("##### the first video first image path: ",  image_paths[0][0]) # "datasets/AVA/frames/-5KQ66BBWC4/-5KQ66BBWC4_000001.jpg"
    print("##### video_idx_to_name lengh: ", len(video_idx_to_name))
    print("##### the first video index: ",  video_idx_to_name[0]) # "-5KQ66BBWC4"
    print("##### ava_helper load_image_lists over!!!")
     
    return image_paths, video_idx_to_name


def load_boxes_and_labels(cfg, mode):
    """
    Loading boxes and labels from csv files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test' mode.
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            [`box_coord`, `box_labels`] where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """
    print("@@@@@@@@@@ ava_helper. load_boxes_and_labels")
    if cfg.TRAIN.USE_SLOWFAST:
        gt_filename = cfg.AVA.TRAIN_GT_BOX_LISTS if mode == 'train' else cfg.AVA.TEST_PREDICT_BOX_LISTS
    else:
        gt_filename = cfg.AVA.TRAIN_GT_BOX_LISTS if mode == 'train' else cfg.AVA.VAL_GT_BOX_LISTS
    ann_filename = os.path.join(cfg.AVA.ANNOTATION_DIR, gt_filename[0])
    all_boxes = {}
    count = 0
    unique_box_count = 0
    if mode == 'train':
        excluded_keys = read_exclusions(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.TRAIN_EXCLUSION_FILE)
        )
    else:
        excluded_keys = read_exclusions(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.EXCLUSION_FILE)
        )
    detect_thresh = cfg.AVA.DETECTION_SCORE_THRESH

    with PathManager.open(ann_filename, 'r') as f:
        for line in f:
            row = line.strip().split(',')

            # use same detection as slowfast
            if cfg.TRAIN.USE_SLOWFAST:
                if mode == 'val' or mode == 'test':
                    score = float(row[7])
                    if score < detect_thresh:
                        continue
            ########

            video_name, frame_sec = row[0], int(row[1])
            key = "%s,%04d" % (video_name, frame_sec)
            # if mode == 'train' and key in excluded_keys:
            if key in excluded_keys:
                print("Found {} to be excluded...".format(key))
                continue

            # Only select frame_sec % 4 = 0 samples for validation if not
            # set FULL_TEST_ON_VAL (default False)
            if mode == 'val' and not cfg.AVA.FULL_TEST_ON_VAL and frame_sec % 4 != 0:
                continue
            # Box with [x1, y1, x2, y2] with a range of [0, 1] as float
            box_key = ",".join(row[2:6])
            box = list(map(float, row[2:6]))
            label = -1 if row[6] == "" else int(row[6])
            if video_name not in all_boxes:
                all_boxes[video_name] = {}
                for sec in AVA_VALID_FRAMES:
                    all_boxes[video_name][sec] = {}
            if box_key not in all_boxes[video_name][frame_sec]:
                all_boxes[video_name][frame_sec][box_key] = [box, []]
                unique_box_count += 1

            all_boxes[video_name][frame_sec][box_key][1].append(label)
            if label != -1:
                count += 1

    for video_name in all_boxes.keys():
        for frame_sec in all_boxes[video_name].keys():
            # Save in format of a list of [box_i, box_i_labels].
            all_boxes[video_name][frame_sec] = list(
                all_boxes[video_name][frame_sec].values()
            )

    logger.info(
        "Finished loading annotations from: %s" % ", ".join([ann_filename])
    )
    logger.info("Number of unique boxes: %d" % unique_box_count)
    logger.info("Number of annotations: %d" % count)
    print("##### len(all_boxes): ", len(all_boxes))
    return all_boxes


def get_keyframe_data(boxes_and_labels):
    """
    Getting keyframe indices, boxes and labels in the dataset.

    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.

    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    def sec_to_frame(sec):
        """
        Convert time index (in second) to frame index.
        0: 900
        30: 901
        """
        return (sec - 900) * FPS

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0
    for video_idx in range(len(boxes_and_labels)):
        sec_idx = 0
        keyframe_boxes_and_labels.append([])
        for sec in boxes_and_labels[video_idx].keys():
            if sec not in AVA_VALID_FRAMES:
                continue

            if len(boxes_and_labels[video_idx][sec]) > 0:
                keyframe_indices.append(
                    (video_idx, sec_idx, sec, sec_to_frame(sec))
                )
                keyframe_boxes_and_labels[video_idx].append(
                    boxes_and_labels[video_idx][sec]
                )
                sec_idx += 1
                count += 1
    logger.info("%d keyframes used." % count)

    return keyframe_indices, keyframe_boxes_and_labels


def get_num_boxes_used(keyframe_indices, keyframe_boxes_and_labels):
    """
    Get total number of used boxes.

    Args:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.

    Returns:
        count (int): total number of used boxes.
    """

    count = 0
    for video_idx, sec_idx, _, _ in keyframe_indices:
        count += len(keyframe_boxes_and_labels[video_idx][sec_idx])
    return count


def get_max_objs(keyframe_indices, keyframe_boxes_and_labels):
    # max_objs = 0
    # for video_idx, sec_idx, _, _ in keyframe_indices:
    #     num_boxes = len(keyframe_boxes_and_labels[video_idx][sec_idx])
    #     if num_boxes > max_objs:
    #         max_objs = num_boxes

    # return max_objs
    return 50 #### MODIFICATION FOR NOW! TODO: FIX LATER!
