# Notes

## 2024-08-29 & before
- Confirmed full body RTMPose inference working
- Setting up coco-wholebody dataset following file structure
- Try training with `configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py`
```bash
python tools/train.py configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py

```
- Try customizing dataset

todos:
- Prepare VEHS-7M into coco format
- Deal with different keypoint seq
- Train test on VEHS-7M
- Merge training
##
