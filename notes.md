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

## Train test with VEHS-7M
- Image frames and VEHS-7M-5fp-37kpts annotation in COCO format ready
- Need to save vicon-read pkl file as json
- Configured:
  - `configs/_base_/datasets/VEHS7M-37kpts.py`
  - `mmpose/datasets/datasets/wholebody/VEHS7M-37kpts_dataset.py`
  - `configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-l_8xb320-270e_VEHS7MOnly-384x288.py`
- Maybe still need to add val bbox file like `data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json`
- **Try training with VEHS7M dataset**
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
- Using a small file to train first, to see if it works
- Error in eval:
  ```
  File "/home/leyang/anaconda3/envs/mmpose/lib/python3.8/site-packages/xtcocotools/cocoeval.py", line 257, in <dictcomp>
    self.ious = {(imgId, catId): computeIoU(imgId, catId) \
  File "/home/leyang/anaconda3/envs/mmpose/lib/python3.8/site-packages/xtcocotools/cocoeval.py", line 373, in computeOks
    dx = xd - xg
  ValueError: operands could not be broadcast together with shapes (37,) (13,3)
  ```
  - Added joint weights and sigmas with the same number of keypoints
  - Seems to be caused by xg read: 37,3 --> 13,3, where should be 37*3,1 --> 37,1
    - keypoints in annotation file should be 1d
  - Resume training
  ```bash
  python tools/train.py configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-l_8xb320-270e_VEHS7MOnly-384x288.py --resume
  ```
- Error:
  ```
    File "/home/leyang/anaconda3/envs/mmpose/lib/python3.8/site-packages/xtcocotools/cocoeval.py", line 382, in computeOks
    e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
  KeyError: 'area'
  ```
    - Need 'area'
    - Area is the area of the segmentation mask [link](https://github.com/cocodataset/cocoapi/issues/36)
    - Just use the bounding box area for now, seems to affect the iou evaluation's small med large size bb
    - Or set COCOeval use_are=False [link](https://mmpose.readthedocs.io/en/latest/_modules/mmpose/evaluation/metrics/coco_metric.html)
- Training on local linux, next step is to move to ARC

## ARC slurm preparation
- Setting up env following [link](https://mmpose.readthedocs.io/en/latest/installation.html)
  - Need python 3.8 it seems, but ARC have 3.10 as default for pytorch.
    - Using conda and installing pytorch with pip, instead of module load
    - Load cuda modules first
    - `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` --> 2.4.1, no good, dont use
  - MMCV version error:
    ```
    MMCV==2.2.0 is used but incompatible.
    ```
    - Tried `mim install "mmcv==2.1.0"`, not working
    - Tried `pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html`
      - only for 1.x, not 2.x
    - `python -c 'import torch;print(torch.__version__)'` --> 2.4.1
    - Local machine --> 2.1.1
    - I think pytorch 2.4 --> mmcv 2.2, but I need mmcv 2.1 from here [link](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip)
       - Uninstall torch stuff
       - CUDA 11.8 - linux `conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia`
       - `pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html`
       - Success
      ```
      Cuda 11.8
      pytorch 2.1.1
      $ pip list | grep mm
      mmcv                   2.1.0
      mmdet                  3.2.0
      mmengine               0.10.5
      mmpose                 1.3.2
      ```
    - Error:
    ```
        ext = importlib.import_module('mmcv.' + name)
    File "/home/wenleyan/.conda/envs/openmmlab/lib/python3.8/importlib/__init__.py", line 127, in import_module
      return _bootstrap._gcd_import(name[level:], package, level)
    ImportError: libc10_cuda.so: cannot open shared object file: No such file or directory
    ```
       - Maybe caused by no GPU on interactive node, try on GPU node --> same error
       - `python -c "import torch; print(torch.__path__[0])"`
       - `ls /home/wenleyan/.conda/envs/openmmlab/lib/python3.8/site-packages/torch/lib` can not find libc10_cuda.so
       - First activate conda, then load cuda module
       - Deleted old and try reinstalling pytorch with conda `conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia`
         - worked, have `libtorch_cuda.so` now
- interactive session not working due to NVIDIA driver issue, try on GPU node
  - GPU node working on batch file, training!!!
- Try full dataset, got CUDA out of memory error
  - I got 3 A40 which have 48GB each
  - 160 batch size still got CUDA out of memory error on 3 GPU
  - On 1 GPU, 32 batch size is using 26% mem, which means I can use 90%/26%*32 = 110 batch size (*3?)
  - Try 115 with 3 GPU  --> 89% mem
- Only 1 GPU training
  - try #SBATCH --ntasks-per-node=3 set to GPU number, still 1 GPU
  - Asking IT: Try `srun python tools/train.py \` & `#SBATCH --gpu-bind=single:1`
  - [MMPose Train w. multi GPU](https://mmpose.readthedocs.io/en/latest/user_guides/train_and_test.html#train-with-multiple-gpus)
  - 
## Prepare other datasets
- COCO WholeBody
- AIC
- MPII-TRB

```bash
python tools/train.py configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-l_8xb320-270e_VEHS7Mplus-384x288.py --resume
```

- First try with VEHS-7M only, but merge, works
- Try adding COCO WholeBody, need to add mapping & converter pipeline in config python file
  - todo: make sure that left and right is correct 
  - Dataset name from `mmpose/datasets/datasets/wholebody/__init__.py`
  - Running
  - Error:
    ```
      File "/home/leyang/Documents/mmpose/mmpose/datasets/transforms/converting.py", line 112, in transform
      c = results[key].shape[-1]
      AttributeError: 'NoneType' object has no attribute 'shape'
    ```
    - kpt mapping conversion caused the error 
    - Tried traininig with COCO-wholebody only using mmpose config, works
    - the coco-wholebody image is 1 based, changing to o 0-based
    - results['keypoints_3d'] key exists, but is None, causing the error, should not exist, and only 'keypoints' should exist
      - Need to find out where it is loaded. 
      - Bug in vicon-read repo that caused the json file to say: num_keypoints=3 instead of 37, maybe that caused the error
      - Final cause: `mmpose/datasets/datasets/wholebody/coco_wholebody_dataset.py` introduced the 'keypoints_3d' key, and set it to None, leading it down the wrong path
  - Working, but killed due to memeory
    - Use 4 batch size, remember to change in ARC and test optimal batch size.
    - Working
- Try AIC, MPII, MPII-TRB
  - Downloaded whole data from [link](https://openxlab.org.cn/datasets/OpenDataLab/AI_Challenger/cli/main) --> good for downloading images and all
  - Download annotation in coco Format in [link](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html?highlight=aic) and put in `/annotations`
  - Modified train config to modify:
    - data root
    - ann file path
    - image prefix path
    - dataset name, according to `mmpose/datasets/datasets/body/__init__.py`
  - MPII & MPII-TRB working, AIC seems to be missing figures, probably due to partial unzip process. --> unzipped again and fixed
  - How is my VEHS-7M sequence chosen?

