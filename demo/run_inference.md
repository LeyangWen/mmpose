## Set Up Docker
link: https://mmpose.readthedocs.io/en/latest/installation.html?highlight=docker
### Build image
```
sudo sudo docker build -t mmpose-inference -f docker/Dockerfile.inference .
```
### start container
```
sudo docker run --gpus all --shm-size=8g --rm -it -v "$PWD":/mmpose -w /mmpose mmpose-inference bash
```
### run inference
```
python demo/topdown_demo_with_mmdet_rtmpose_for_Gunwoo.py \
    demo/mmdetection_cfg/rtmdet_nano_320-8xb32_coco-person.py \
    demo/checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-x_8xb320-270e_VEHS7Mplus-384x288_exp_2b.py \
    demo/checkpoints/RTMW37kpts_v2_20fps-finetune-pitch-correct-5-angleLossV2-only/best_epoch.bin \
    --input demo/resources/exp_1/lumber_lift.mp4 \
    --output-root demo/resources/exp_1/output \
    --frames_folder demo/resources/exp_1/frames \
    --fps 20 \
```


```
Loads checkpoint by local backend from path: demo/checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth
/opt/conda/lib/python3.7/site-packages/torch/cuda/init.py:104: UserWarning:
NVIDIA GeForce RTX 3060 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_61 sm_70 sm_75 compute_37.
If you want to use the NVIDIA GeForce RTX 3060 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

warnings.warn(incompatible_device_warn.format(device_name, capability, " ".join(arch_list), device_name))
```
