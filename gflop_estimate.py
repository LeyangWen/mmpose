import torch
from mmcv.cnn import get_model_complexity_info
from mmengine.config import Config
from mmpose.models import build_pose_estimator
from mmengine.runner import load_checkpoint
from mmpose.apis import inference_topdown, init_model



# --- your existing helper ---------------------------------------------------
def get_gflops(module, in_shape):
    flops, _ = get_model_complexity_info(
        module, in_shape, as_strings=False,
        print_per_layer_stat=False)
    return flops / 1e9

def compare_module_weights(module1, module2, module_name):
    print(f"\nComparing weights for module: {module_name}")
    m1 = getattr(module1, module_name)
    m2 = getattr(module2, module_name)
    for (name1, param1), (name2, param2) in zip(m1.named_parameters(), m2.named_parameters()):
        if not torch.equal(param1.data, param2.data):
            diff = (param1 - param2).abs().mean().item()
            print(f"  {name1}: DIFFERENT, mean abs diff = {diff:.6f}")
            
        else:
            print(f"  {name1}: ------------------------ Identical")
        # print(param1.data[:5])  # Print first 5 values for comparison
    print("#"* 50)

# --- 1) build model ---------------------------------------------------------

# cfg = Config.fromfile('configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py')
# cfg = Config.fromfile('configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py')
# cfg = Config.fromfile('/home/leyang/Documents/mmpose/configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-l_8xb320-270e_VEHS7Mplus-384x288.py')
cfg = Config.fromfile('/home/leyang/Documents/mmpose/configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-x_8xb320-270e_VEHS7Mplus-384x288_exp_1a.py')

model = init_model(cfg, device='cuda:0')
                   
checkpoint = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/my_exp_1_freeze/20250607_010951/epoch_1.pth'
# checkpoint = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/my_exp_1_freeze/20250607_010452/epoch_1.pth'
# checkpoint = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/my_exp_2_unfreeze/rtmw-l_8xb320-270e_VEHS7Mplus-384x288_1/epoch_1.pth'
# checkpoint = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/my_exp_2_unfreeze/40epoch/epoch_10.pth'
# checkpoint = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/my_exp_1_freeze/40epoch/epoch_10.pth'
# checkpoint = '/home/leyang/Documents/mmpose/work_dirs/rtmw-l_8xb320-270e_VEHS7Mplus-384x288/epoch_1.pth'
# checkpoint = 'work_dirs/rtmw-l_8xb320-270e_VEHS7Mplus-384x288/epoch_1.pth'


# checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/my_exp_1_freeze/20250607_010951/epoch_10.pth'
# checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/my_exp_1_freeze/20250607_010452/epoch_1.pth'
# checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/Experiment#1/best_coco_AP_epoch_270.pth'
# checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/Experiment#2/epoch_270.pth'
# checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/my_exp_2_unfreeze/rtmw-l_8xb320-270e_VEHS7Mplus-384x288_1/epoch_10.pth'
# checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/my_exp_2_unfreeze/40epoch/epoch_40.pth'
# checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/my_exp_1_freeze/40epoch/epoch_40.pth'
# checkpoint_2 = 'work_dirs/rtmw-l_8xb320-270e_VEHS7Mplus-384x288/epoch_10.pth'

# 1a
# checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/exp_1a_1b/exp_1a/best_coco_AP_epoch_130.pth'
checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/exp_1a_1b/exp_1a/epoch_3.pth'

# # 1b
# checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/exp_1a_1b/exp_1b/best_coco_AP_epoch_130.pth'
checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/exp_1a_1b/exp_1b/epoch_3.pth'

# 1c
checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/exp_1a_1b/exp_1c/best_coco_AP_epoch_150.pth'
# checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/exp_1a_1b/exp_1c/epoch_3.pth'


# 2a
checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/exp_1a_1b/exp_2a/best_coco_AP_epoch_30.pth'
# checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/exp_1a_1b/exp_2a/epoch_3.pth'


# 2b
checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/exp_1a_1b/exp_2b/best_coco_AP_epoch_30.pth'
# checkpoint_2 = '/home/leyang/Downloads/Checkpoints_and_log_files_RTMPose_training/exp_1a_1b/exp_2b/epoch_3.pth'


model_2 = init_model(cfg, device='cuda:0', checkpoint=checkpoint_2)
model = init_model(cfg, device='cuda:0', checkpoint=checkpoint)


compare_module_weights(model, model_2, "backbone")
compare_module_weights(model, model_2, "neck")
compare_module_weights(model, model_2, "head")



# cfg = Config.fromfile('/home/leyang/Documents/mmpose/configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-l_8xb320-270e_VEHS7Mplus-384x288.py')


model = init_model(cfg, device='cuda:0')
meta = load_checkpoint(
    model,
    'https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth',
    # 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth',
    # 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth',
    map_location='cuda:0',
    strict=False  # allow you to see missing/unexpected keys
)

cfg_2 = Config.fromfile('/home/leyang/Documents/mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py')
    # '/home/leyang/Documents/mmpose/configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-l_8xb320-270e_VEHS7Mplus-384x288.py')
model_2 = init_model(cfg_2, device='cuda:0')
meta_2 = load_checkpoint(
    model_2,
    # 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth',
    'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth',
    # 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth',
    map_location='cuda:0',
    strict=False  # allow you to see missing/unexpected keys
)

cfg_3 = Config.fromfile('/home/leyang/Documents/mmpose/configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-x_8xb320-270e_VEHS7Mplus-384x288.py')
model_3 = init_model(cfg_3, device='cuda:0')
meta_3 = load_checkpoint(
    model_3,
    'https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth',
    map_location='cuda:0',
    strict=False  # allow you to see missing/unexpected keys
)


model = model_3
####################### input output shape
model.eval()
model = model.cpu()
x = torch.randn(1, 3, 288, 384)
# x = torch.randn(1, 3, 192, 256)

# 1) Backbone
backbone_feats = model.backbone(x)
for i, feat in enumerate(backbone_feats):
    print(f'backbone output {i:>2d}: {tuple(feat.shape)}')

# 2) Neck
neck_feats = model.neck(backbone_feats)  # pass only the in_channels levels
for i, feat in enumerate(neck_feats):
    print(f'  neck output {i}: {tuple(feat.shape)}')

# 3) Head
# if head takes multiple neck outputs you can adapt accordingly
# often head just takes the last feature-level:
y = model.head(neck_feats)
for i, feat in enumerate(y):
    print(f'  head output {i:>2d}: {tuple(feat.shape)}')
    




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
