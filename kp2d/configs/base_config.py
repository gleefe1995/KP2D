# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""Default kp2d configuration parameters (overridable in configs/*.yaml)
"""

import os
from yacs.config import CfgNode as CN

########################################################################################################################
cfg = CN()
cfg.name = ''       # Run name
cfg.debug = True   # Debugging flag
########################################################################################################################
### ARCH
########################################################################################################################
cfg.arch = CN()
cfg.arch.seed = 42                  # Random seed for Pytorch/Numpy initialization
cfg.arch.epochs = 50                # Maximum number of epochs
########################################################################################################################
### WANDB
########################################################################################################################
cfg.wandb = CN()
cfg.wandb.dry_run = True                                 # Wandb dry-run (not logging)
cfg.wandb.name = ''                                      # Wandb run name
cfg.wandb.project = os.environ.get("WANDB_PROJECT", "")  # Wandb project
cfg.wandb.entity = os.environ.get("WANDB_ENTITY", "")    # Wandb entity
cfg.wandb.tags = []                                      # Wandb tags
cfg.wandb.dir = ''                                       # Wandb save folder
########################################################################################################################
### MODEL
########################################################################################################################
cfg.model = CN()
cfg.model.checkpoint_path = '/home/hongbeom/kp2d/data/experiments/kp2d/'              # Checkpoint path for model saving
cfg.model.save_checkpoint = True
########################################################################################################################
### MODEL.SCHEDULER
########################################################################################################################
cfg.model.scheduler = CN()
cfg.model.scheduler.decay = 0.5                                # Scheduler decay rate
cfg.model.scheduler.lr_epoch_divide_frequency = 40             # Schedule number of epochs when to decay the initial learning rate by decay rate
########################################################################################################################
### MODEL.OPTIMIZER
########################################################################################################################
cfg.model.optimizer = CN()
cfg.model.optimizer.learning_rate = 0.001
cfg.model.optimizer.weight_decay = 0.0
########################################################################################################################
### MODEL.PARAMS
########################################################################################################################
cfg.model.params = CN()                                     
cfg.model.params.keypoint_loss_weight = 1.0                 # Keypoint loss weight
cfg.model.params.descriptor_loss_weight = 1.0               # Descriptor loss weight
cfg.model.params.score_loss_weight = 1.0                    # Score loss weight
cfg.model.params.use_color = True                           # Use color or grayscale images
cfg.model.params.with_io = True                             # Use IONet
cfg.model.params.do_upsample = True                         # Upsample descriptors
cfg.model.params.do_cross = True                            # Use cross-border keypoints
cfg.model.params.descriptor_loss = True                     # Use hardest negative mining descriptor loss
cfg.model.params.keypoint_net_type = 'KeypointNet'          # Type of keypoint network. Supported ['KeypointNet', 'KeypointResnet']
########################################################################################################################
### DATASETS
########################################################################################################################
cfg.datasets = CN()
########################################################################################################################
### DATASETS.AUGMENTATION
########################################################################################################################
cfg.datasets.augmentation = CN()
cfg.datasets.augmentation.image_shape = (240, 320)              # Image shape
cfg.datasets.augmentation.jittering = (0.5, 0.5, 0.2, 0.05)     # Color jittering values
########################################################################################################################
### DATASETS.TRAIN
########################################################################################################################
cfg.datasets.train = CN()
cfg.datasets.train.batch_size = 8                               # Training batch size
cfg.datasets.train.num_workers = 16                                    # Training number of workers
cfg.datasets.train.path = '/data/hongbeom/Downloads/train2017'        # Training data path (COCO dataset)
cfg.datasets.train.repeat = 1                                          # Number of times training dataset is repeated per epoch
########################################################################################################################
### DATASETS.VAL
########################################################################################################################
cfg.datasets.val = CN()
cfg.datasets.val.path = '/data/hongbeom/Downloads/hpatches-sequences-release'     # Validation data path (HPatches)
########################################################################################################################
### THESE SHOULD NOT BE CHANGED
########################################################################################################################
cfg.config = ''                 # Run configuration file
cfg.default = ''                # Run default configuration file
cfg.wandb.url = ''              # Wandb URL
########################################################################################################################
# pose_resnet related params
# POSE_RESNET = CN()
# POSE_RESNET.NUM_LAYERS = 50
# POSE_RESNET.DECONV_WITH_BIAS = False
# POSE_RESNET.NUM_DECONV_LAYERS = 3
# POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
# POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
# POSE_RESNET.FINAL_CONV_KERNEL = 1
# POSE_RESNET.PRETRAINED_LAYERS = ['*']

# pose_multi_resoluton_net related params
# cfg = CN()
cfg.PRETRAINED_LAYERS = ['*']
cfg.STEM_INPLANES = 64
cfg.FINAL_CONV_KERNEL = 1

cfg.STAGE2 = CN()
cfg.STAGE2.NUM_MODULES = 1
cfg.STAGE2.NUM_BRANCHES = 2
cfg.STAGE2.NUM_BLOCKS = [4, 4]
cfg.STAGE2.NUM_CHANNELS = [32, 64]
cfg.STAGE2.BLOCK = 'BASIC'
cfg.STAGE2.FUSE_METHOD = 'SUM'

cfg.STAGE3 = CN()
cfg.STAGE3.NUM_MODULES = 1
cfg.STAGE3.NUM_BRANCHES = 3
cfg.STAGE3.NUM_BLOCKS = [4, 4, 4]
cfg.STAGE3.NUM_CHANNELS = [32, 64, 128]
cfg.STAGE3.BLOCK = 'BASIC'
cfg.STAGE3.FUSE_METHOD = 'SUM'

cfg.STAGE4 = CN()
cfg.STAGE4.NUM_MODULES = 1
cfg.STAGE4.NUM_BRANCHES = 4
cfg.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
cfg.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
cfg.STAGE4.BLOCK = 'BASIC'
cfg.STAGE4.FUSE_METHOD = 'SUM'


# MODEL_EXTRAS = {
    
#     'pose_high_resolution_net': POSE_HIGH_RESOLUTION_NET,
# }




cfg.OUTPUT_DIR = ''
cfg.LOG_DIR = ''
cfg.DATA_DIR = ''
cfg.GPUS = (0,)
cfg.WORKERS = 4
cfg.PRINT_FREQ = 20
cfg.AUTO_RESUME = False
cfg.PIN_MEMORY = True
cfg.RANK = 0

# Cudnn related params
cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

# common params for NETWORK
cfg.MODEL = CN()
cfg.MODEL.NAME = 'pose_hrnet'
cfg.MODEL.INIT_WEIGHTS = True
cfg.MODEL.PRETRAINED = ''
cfg.MODEL.NUM_JOINTS = 17
cfg.MODEL.TAG_PER_JOINT = True
cfg.MODEL.TARGET_TYPE = 'gaussian'
cfg.MODEL.IMAGE_SIZE = [240, 320]  # width * height, ex: 192 * 256
cfg.MODEL.HEATMAP_SIZE = [60, 80]  # width * height, ex: 24 * 32
cfg.MODEL.SIGMA = 2
cfg.MODEL.EXTRA = CN(new_allowed=True)

cfg.LOSS = CN()
cfg.LOSS.USE_OHKM = False
cfg.LOSS.TOPK = 8
cfg.LOSS.USE_TARGET_WEIGHT = True
cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
cfg.DATASET = CN()
cfg.DATASET.ROOT = ''
cfg.DATASET.DATASET = 'mpii'
cfg.DATASET.TRAIN_SET = 'train'
cfg.DATASET.TEST_SET = 'valid'
cfg.DATASET.DATA_FORMAT = 'jpg'
cfg.DATASET.HYBRID_JOINTS_TYPE = ''
cfg.DATASET.SELECT_DATA = False

# training data augmentation
cfg.DATASET.FLIP = True
cfg.DATASET.SCALE_FACTOR = 0.25
cfg.DATASET.ROT_FACTOR = 30
cfg.DATASET.PROB_HALF_BODY = 0.0
cfg.DATASET.NUM_JOINTS_HALF_BODY = 8
cfg.DATASET.COLOR_RGB = False

# train
cfg.TRAIN = CN()

cfg.TRAIN.LR_FACTOR = 0.1
cfg.TRAIN.LR_STEP = [90, 110]
cfg.TRAIN.LR = 0.001

cfg.TRAIN.OPTIMIZER = 'adam'
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WD = 0.0001
cfg.TRAIN.NESTEROV = False
cfg.TRAIN.GAMMA1 = 0.99
cfg.TRAIN.GAMMA2 = 0.0

cfg.TRAIN.BEGIN_EPOCH = 0
cfg.TRAIN.END_EPOCH = 140

cfg.TRAIN.RESUME = False
cfg.TRAIN.CHECKPOINT = ''

cfg.TRAIN.BATCH_SIZE_PER_GPU = 4
cfg.TRAIN.SHUFFLE = True

# testing
cfg.TEST = CN()

# size of images for each device
cfg.TEST.BATCH_SIZE_PER_GPU = 4
# Test Model Epoch
cfg.TEST.FLIP_TEST = False
cfg.TEST.POST_PROCESS = False
cfg.TEST.SHIFT_HEATMAP = False

cfg.TEST.USE_GT_BBOX = False

# nms
cfg.TEST.IMAGE_THRE = 0.1
cfg.TEST.NMS_THRE = 0.6
cfg.TEST.SOFT_NMS = False
cfg.TEST.OKS_THRE = 0.5
cfg.TEST.IN_VIS_THRE = 0.0
cfg.TEST.COCO_BBOX_FILE = ''
cfg.TEST.BBOX_THRE = 1.0
cfg.TEST.MODEL_FILE = ''

# debug
cfg.DEBUG = CN()
cfg.DEBUG.DEBUG = False
cfg.DEBUG.SAVE_BATCH_IMAGES_GT = False
cfg.DEBUG.SAVE_BATCH_IMAGES_PRED = False
cfg.DEBUG.SAVE_HEATMAPS_GT = False
cfg.DEBUG.SAVE_HEATMAPS_PRED = False





def get_cfg_defaults():
    return cfg.clone()
