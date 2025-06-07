import torch
from mmcv.cnn import get_model_complexity_info
from mmengine.config import Config
from mmpose.models import build_pose_estimator

from mmpose.apis import inference_topdown, init_model



# --- your existing helper ---------------------------------------------------
def get_gflops(module, in_shape):
    flops, _ = get_model_complexity_info(
        module, in_shape, as_strings=False,
        print_per_layer_stat=False)
    return flops / 1e9

# --- 1) build model ---------------------------------------------------------

cfg = Config.fromfile('configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py')
# cfg = Config.fromfile('configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py')
# cfg = Config.fromfile('/home/leyang/Documents/mmpose/configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-l_8xb320-270e_VEHS7Mplus-384x288.py')
model = init_model(cfg, device='cuda:0')
# --- 2) measure backbone & backbone+neck (you already have this) -------------
gflops_bb    = get_gflops(model.backbone, (3, 384, 288))
# BBNeck wrapper from before
class BBNeck(torch.nn.Module):
    def __init__(self, bb, neck):
        super().__init__()
        self.bb, self.neck = bb, neck
    def forward(self, x):
        feats = self.bb(x)       
        return self.neck(feats)  

wrapper_bn      = BBNeck(model.backbone, model.neck).cuda().eval()
gflops_bb_neck  = get_gflops(wrapper_bn, (3, 384, 288))

# --- 3) wrap backbone+neck+head ---------------------------------------------
class BBNeckHead(torch.nn.Module):
    def __init__(self, bb, neck, head):
        super().__init__()
        self.bb   = bb
        self.neck = neck
        self.head = head
    def forward(self, x):
        feats = self.bb(x)          # tuple of tensors
        neck  = self.neck(feats)    # single tensor
        return self.head(neck)      # heatmap tensor

wrapper_bnh     = BBNeckHead(model.backbone, model.neck, model.head).cuda().eval()
gflops_bb_neck_head = get_gflops(wrapper_bnh, (3, 384, 288))

# --- 4) subtract to get head alone ------------------------------------------
gflops_head = gflops_bb_neck_head - gflops_bb_neck

print(f"Backbone      : {gflops_bb:.2f} G")
print(f"Backbone+Neck : {gflops_bb_neck:.2f} G")
print(f"Backbone+Neck+Head: {gflops_bb_neck_head:.2f} G")
print(f"→   Neck      : {gflops_bb_neck - gflops_bb:.2f} G")
print(f"→   Head      : {gflops_head:.2f} G")




meta = load_checkpoint(
    model,
    # 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth',
    'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth',
    map_location='cuda:0',
    logger=logging.getLogger('checkpoint_test'),
    strict=False  # allow you to see missing/unexpected keys
)