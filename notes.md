# Notes

## 2024-08-29 & before
- Confirmed full body RTMPose inference working
- Setting up coco-wholebody dataset following file structure
- Try training with `configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py`
```bash
python tools/train.py configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py
```
- Confirmed working 
- Try customizing dataset

todos:
- Prepare VEHS-7M into coco format
- Deal with different keypoint seq
- Train test on VEHS-7M
- Merge training test
- Move to slurm

## 2024-09-03 notes
- Running inference again
```bash
python demo/image_demo.py \
    test_1.png \
    configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py \
    rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth \
    --out-file vis_results.jpg
    
```

## 2024-10-03 notes
- Image frames and VEHS-7M-5fp-37kpts annotation in COCO format ready
- Need to save vicon-read pkl file as json
- Configured:
  - `configs/_base_/datasets/VEHS7M-37kpts.py`
  - `mmpose/datasets/datasets/wholebody/VEHS7M-37kpts_dataset.py`
  - `configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-l_8xb320-270e_VEHS7MOnly-384x288.py`
- Maybe still need to add val bbox file like `data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json`
- Try training with VEHS7M dataset
```bash
python tools/train.py configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-l_8xb320-270e_VEHS7MOnly-384x288.py
```

- Error:
    ```
    KeyError: 'VEHS7M37kptsDataset is not in the mmpose::dataset registry.
    ```
  - Need to modify init file `mmpose/datasets/datasets/wholebody/__init__.py`
  - Fixed
- Error:
    ```
    json.decoder.JSONDecodeError: Expecting value: line 1 column 37009362 (char 37009361)
    ```
  - Seems to be memory issue
  - Turns out that vicon-read --> json file is quite hard due to lots of high dim arrays, np saves memory, but json needs list
  - Modified the json file generation code to write iteratively, 50 min write for each sesssion, 90 in total --> 90 hours ish
  - Writing to ssd is much faster 10 min per session, 18 hours in total
  - Turns out to be a bug in bbox write, which causes it to write many times, fixed, and now generation is instant

- Error:
    ```
      File "<__array_function__ internals>", line 200, in concatenate
    ValueError: need at least one array to concatenate
    ```
  - Can be caused by many things, in my case, I used 'joint_2d' key instead of 'keypoints' for 2D pose
  - Caused in `mmpose/datasets/datasets/base/base_coco_style_dataset.py`-`def parse_data_info`, resulting in empty `ann_info`
  - Need to fix in Vicon-read
    - `joint_2d` should be 'keypoints'
    - 'bbox' should be in xy(tl)wh format, now in xytl xybr
    - 62 repetitive cases in train, 9 cases in test where `"id": 1, "image_id": 1},`, different joint 2ds
    - Downsample mismatch between img and annotation: Frame count need to restart for each camera, and start at 1, middle frame is sampled by ffmpeg
  - Fixed, now running training
  
- Error:
    ```
    [1]    521981 killed     python tools/train.py 
    ```
  - Seems to be memory issue, confirm by `sudo dmesg | grep -i 'killed process'`
  - Try to reduce batch size to 4
  - Todo: remember to change back config python in ARC
- Error:
  ```
    File "/home/leyang/Documents/mmpose/mmpose/models/losses/classification_loss.py", line 211, in forward
      t_loss[:, self.mask] = t_loss[:, self.mask] * self.mask_weight
  
  IndexError: index 37 is out of bounds for dimension 0 with size 37
  ```
  - Caused by loss mask in python cofig, fix:
    ```python
    loss=dict(
        type='KLDiscretLoss',
        use_target_weight=True,
        beta=1.,
        label_softmax=True,
        label_beta=10.,
        # mask=list(range(23, 91)),  # for full body face keypoints in cocktail14 merge training
        # mask_weight=0.5,
    ),
    ```
- Training on local linux, next step is to move to ARC