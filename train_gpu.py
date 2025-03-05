import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
import pickle
import torch.nn.functional as F
from sklearn.metrics import classification_report , precision_recall_fscore_support
import iopath
import simplejson
import psutil
from torchmetrics import F1Score
import os
import math
from functools import partial
from mvit.models.attention import MultiScaleBlock
from mvit.models.common import round_width
from mvit.utils.misc import validate_checkpoint_wrapper_import
#from torch.nn.init import trunc_normal_
try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None
import argparse
import sys
import mvit.utils.checkpoint as cu
from tools.engine import test, train
from mvit.config.defaults import assert_and_infer_cfg, get_cfg
from mvit.utils.misc import launch_job
from mvit.models import build_model
from fvcore.common.registry import Registry
import mvit.utils.distributed as du
from datetime import datetime
import time
import logging
logger = logging.getLogger(__name__)
from log import setupLogging
from mvit.utils.env import checkpoint_pathmgr as pathmgr
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from fvcore.common.registry import Registry
import pprint
import mvit.models.losses as losses
import mvit.models.optimizer as optim
import mvit.utils.checkpoint as cu
import mvit.utils.distributed as du
import mvit.utils.metrics as metrics
import mvit.utils.misc as misc
from mvit.datasets import loader
from mvit.datasets.mixup import MixUp
from mvit.models import build_model
from mvit.utils.meters import EpochTimer, TrainMeter, ValMeter
import configparser
from pathlib import Path
import shutil
from evaluate import getPerformance, predict_val, printEpochWise, plotElbow

##>>>>>>>>>>>>>>>>>> CONFIGURATION SETUP HERE >>>>>>>>>>>>>>>>>>>>>
# location of pretrained modelget_cfg
# Loading pre-trained model weights into current model

### LF:setting up so that the load path is the same as the save path
## please be aware that you are training/overwriting the same file now - if you would like to make a backup of the current state, please make a COPY into a different location
cfg = get_cfg() # get_cfg = init the config being used for the entire project


class MyConfigParser(configparser.RawConfigParser):
    def get(self, section, option):
        val = configparser.RawConfigParser.get(self, section, option)
        return val.strip('"').strip("'")

config_obj = MyConfigParser()
# check if config.ini is available, else make a copy of config.ini.bak and rename it to config.ini
config_path = "./config.ini"
if not Path(config_path).is_file():
    shutil.copyfile("config.ini.bak","config.ini")
config_obj.read(config_path)

## if you haven't trained before, the initial state is just the pretrained model weights
cfg.OUTPUT_DIR = config_obj.get("MODEL","OUTPUT_DIR") # can change this if you like 
model_name = config_obj.get("MODEL","MODEL_NAME") # can change this if you like if you change models
my_checkpoint_path = cfg.OUTPUT_DIR + '/' + model_name # don't change this 

cfg.MVIT.POOL_KV_STRIDE =[]
# T model settings
cfg.MVIT.DROPPATH_RATE= 0.1
cfg.MVIT.DEPTH= 10
cfg.MVIT.DIM_MUL= [[1, 2.0], [3, 2.0], [8, 2.0]]
cfg.MVIT.HEAD_MUL= [[1, 2.0], [3, 2.0], [8, 2.0]]
cfg.MVIT.POOL_KVQ_KERNEL= [3, 3]
cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE= [4, 4]
cfg.MVIT.POOL_Q_STRIDE= [[0, 1, 1], [1, 2, 2], [2, 1, 1], [3, 2, 2], [4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 2, 2], [9, 1, 1]]
# B model settings
# cfg.MVIT.DROPPATH_RATE= 0.3
# cfg.MVIT.DEPTH= 24
# cfg.MVIT.DIM_MUL= [[2, 2.0], [5, 2.0], [21, 2.0]]
# cfg.MVIT.HEAD_MUL= [[2, 2.0], [5, 2.0], [21, 2.0]]
# cfg.MVIT.POOL_KVQ_KERNEL= [3, 3]
# cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE= [4, 4]
# cfg.MVIT.POOL_Q_STRIDE= [[0, 1, 1], [1, 1, 1], [2, 2, 2], [3, 1, 1], [4, 1, 1], [5, 2, 2], [6, 1, 1], [7, 1, 1], [8, 1, 1], [9, 1, 1], [10, 1, 1], [11, 1, 1], [12, 1, 1], [13, 1, 1], [14, 1, 1], [15, 1, 1], [16, 1, 1], [17, 1, 1], [18, 1, 1], [19, 1, 1], [20, 1, 1], [21, 2, 2], [22, 1, 1], [23, 1, 1]]

# Solver settings are the same for T and B models
cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS= True
cfg.SOLVER.BASE_LR= 0.00025
cfg.SOLVER.LR_POLICY= 'cosine'
cfg.SOLVER.WEIGHT_DECAY= 0.05
cfg.SOLVER.OPTIMIZING_METHOD= 'adamw'
cfg.SOLVER.CLIP_GRAD_L2NORM= 1.0

#### m1 mac settings, mainly for LF - disabling
# try:
#     torch.has_mps
#     if torch.has_mps:
#         use_mac = 1
#     else:
#         use_mac=0
# except:
#     use_mac = 0
use_mac=0

##### GENERALLY SPEAKING BELOW ARE THE MAIN STUFF TO CONFIGURE
if torch.cuda.is_available(): 
    cfg['NUM_GPUS'] = 1
else:
    cfg['NUM_GPUS'] = 0
if use_mac == 1:
    device = torch.device("mps")    
if torch.cuda.is_available():
    cfg.DATA_LOADER.NUM_WORKERS = 2 # we're setting this to 0 so that it doesn't use multi-processing, but i'm not sure if this is problematic for GPU...
else:
    cfg.DATA_LOADER.NUM_WORKERS = 0
cfg.MODEL.NUM_CLASSES = int(config_obj.get("MODEL","NUM_CLASSES"))
cfg.DATA.PATH_TO_DATA_DIR = config_obj.get("DATA","PATH_TO_DATA_DIR")
cfg.TRAIN.BATCH_SIZE = int(config_obj.get("TRAIN","BATCH_SIZE"))
cfg.TEST.BATCH_SIZE = int(config_obj.get("TEST","BATCH_SIZE"))
# set epoch to train to
cfg.SOLVER.MAX_EPOCH = int(config_obj.get("SOLVER","MAX_EPOCH"))
cfg.SOLVER.WARMUP_EPOCHS = int(config_obj.get("WARMUP","WARMUP_EPOCHS")) # set number of warmups epochs - defaults to 20 
# seems like certain epochs are evaluation epochs?
cfg.TRAIN.AUTO_RESUME = False
cfg.MIXUP.ENABLE = False
my_layers_frozen = config_obj.get("FREEZE","BLOCKS_FROZEN") # can change this if you like 
# adding data augmentation configurations
if config_obj.get("DATA_AUG","COLOR_JITTER") == "None":
    cfg.AUG.COLOR_JITTER = None
else:
    cfg.AUG.COLOR_JITTER = float(config_obj.get("DATA_AUG","COLOR_JITTER"))

if config_obj.get("DATA_AUG","RAND_AUG") == 'None':
    cfg.AUG.AA_TYPE = None
else:
    cfg.AUG.AA_TYPE = config_obj.get("DATA_AUG","RAND_AUG")
cfg.AUG.RE_PROB = float(config_obj.get("DATA_AUG","ERASE"))
cfg.AUG.NUM_SAMPLE = int(config_obj.get("DATA_AUG","NUM_COUNT"))
if config_obj.get("DATA_AUG","MIXUP") == 'False':
    cfg.MIXUP.ENABLE = False # default is false
else:
    cfg.MIXUP.ENABLE = True


# <<<<<<<<<<<<<<<< END OF CONFIGRATION SETUP HERE <<<<<<<<<<<<<<<<<<#


MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for models.
The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""

def build_model(cfg, gpu_id=None):
    """
    Builds the model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in mvit/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    if torch.has_mps:
        model = model.to(device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
    if use_mac==1:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=device, output_device=device
        )
    return model 


def parse_args():
    """
    Parse the following arguments for a default parser.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/MVIT_B.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See mvit/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

## MAYBE CAN DELETE : load_config function
def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=768,
        kernel=(7, 7),
        stride=(4, 4),
        padding=(3, 3),
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.proj(x.to())
        # B C H W -> B HW C
        return x.flatten(2).transpose(1, 2), x.shape


class TransformerBasicHead(nn.Module):
    """
    Basic Transformer Head. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)
        
        # LF - removing softmax activation function which makes model break in evaluation
        #if not self.training:
        #    x = self.act(x)
        return x


@MODEL_REGISTRY.register()
class MViT(nn.Module):
    """
    Improved Multiscale Vision Transformers for Classification and Detection
    Yanghao Li*, Chao-Yuan Wu*, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2112.01526
    Multiscale Vision Transformers
    Haoqi Fan*, Bo Xiong*, Karttikeya Mangalam*, Yanghao Li*, Zhicheng Yan, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        # Prepare input.
        in_chans = 3
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # MViT params.
        num_heads = cfg.MVIT.NUM_HEADS
        depth = cfg.MVIT.DEPTH
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.zero_decay_pos_cls = cfg.MVIT.ZERO_DECAY_POS_CLS

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if cfg.MODEL.ACT_CHECKPOINT:
            validate_checkpoint_wrapper_import(checkpoint_wrapper)

        patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
        )
        if cfg.MODEL.ACT_CHECKPOINT:
            patch_embed = checkpoint_wrapper(patch_embed)
        self.patch_embed = patch_embed

        patch_dims = [
            spatial_size // cfg.MVIT.PATCH_STRIDE[0],
            spatial_size // cfg.MVIT.PATCH_STRIDE[1],
        ]
        num_patches = math.prod(patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, cfg.MVIT.DROPPATH_RATE, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))

        # MViT backbone configs
        dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv = _prepare_mvit_configs(
            cfg
        )

        input_size = patch_dims
        self.blocks = nn.ModuleList()
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            if cfg.MVIT.DIM_MUL_IN_ATT:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=cfg.MVIT.MLP_RATIO,
                qkv_bias=cfg.MVIT.QKV_BIAS,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=cfg.MVIT.MODE,
                has_cls_embed=self.cls_embed_on,
                pool_first=cfg.MVIT.POOL_FIRST,
                rel_pos_spatial=cfg.MVIT.REL_POS_SPATIAL,
                rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
            )

            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride for size, stride in zip(input_size, stride_q[i])
                ]
            embed_dim = dim_out

        self.norm = norm_layer(embed_dim)

        self.head = TransformerBasicHead(
            embed_dim,
            num_classes,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )
        if self.use_abs_pos:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.zero_decay_pos_cls:
            # add all potential params
            names = ["pos_embed", "rel_pos_h", "rel_pos_w", "cls_token"]

        return names

    def forward(self, x):
        x, bchw = self.patch_embed(x)

        H, W = bchw[-2], bchw[-1]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            x = x + self.pos_embed

        thw = [H, W]
        for blk in self.blocks:
            x, thw = blk(x, thw)

        x = self.norm(x)

        #if self.cls_embed_on:
        #    x = x[:, 0]
        #else:
        #    x = x.mean(1)
#
        x = self.head(x)
        return x


def _prepare_mvit_configs(cfg):
    """
    Prepare mvit configs for dim_mul and head_mul facotrs, and q and kv pooling
    kernels and strides.
    """
    depth = cfg.MVIT.DEPTH
    dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
    for i in range(len(cfg.MVIT.DIM_MUL)):
        dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
    for i in range(len(cfg.MVIT.HEAD_MUL)):
        head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

    pool_q = [[] for i in range(depth)]
    pool_kv = [[] for i in range(depth)]
    stride_q = [[] for i in range(depth)]
    stride_kv = [[] for i in range(depth)]

    for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
        stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][1:]
        pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL

    # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
    if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
        _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
        cfg.MVIT.POOL_KV_STRIDE = []
        for i in range(cfg.MVIT.DEPTH):
            if len(stride_q[i]) > 0:
                _stride_kv = [
                    max(_stride_kv[d] // stride_q[i][d], 1)
                    for d in range(len(_stride_kv))
                ]
            cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

    for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
        stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[i][1:]
        pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL

    return dim_mul, head_mul, pool_q, pool_kv, stride_q,  stride_kv



def load_checkpoint(
    path_to_checkpoint,
    model,
    data_parallel=False,
    optimizer=None,
    scaler=None,
    epoch_reset=False,
    squeeze_temporal=False,
):
    """
    Load the checkpoint from the given file.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
        scaler (GradScaler): GradScaler to load the mixed precision scale.
        epoch_reset (bool): if True, reset #train iterations from the checkpoint.
        squeeze_temporal (bool): if True, squeeze temporal dimension for 3D conv to
            2D conv.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    assert pathmgr.exists(path_to_checkpoint), "Checkpoint '{}' not found".format(
        path_to_checkpoint
    )

# =============== ADDED BY CS10-2: Minjun Jung BEGIN *************** #
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    if use_mac==1: 
        model = model.to(device)
# =============== ADDED BY CS10-2: Minjun Jung END =============== #

    logger.info("Loading network weights from {}.".format(path_to_checkpoint))

    # Account for the DDP wrapper in the multi-gpu setting.
    ms = model.module if data_parallel else model

    # Load the checkpoint on CPU to avoid GPU mem spike.
    with pathmgr.open(path_to_checkpoint, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    pre_train_dict = checkpoint["model_state"]
    model_dict = ms.state_dict()

    if squeeze_temporal:
        for k, v in pre_train_dict.items():
            # convert 3D conv to 2D
            if (
                k in model_dict
                and len(v.size()) == 5
                and len(model_dict[k].size()) == 4
                and v.size()[2] == 1
            ):
                pre_train_dict[k] = v.squeeze(2)

    # Match pre-trained weights that have same shape as current model.
    pre_train_dict_match = {
        k: v
        for k, v in pre_train_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }
    # Weights that do not have match from the pre-trained model.
    not_load_layers = [
        k for k in model_dict.keys() if k not in pre_train_dict_match.keys()
    ]
    # Log weights that are not loaded with the pre-trained weights.
    if not_load_layers:
        for k in not_load_layers:
            logger.info("Network weights {} not loaded.".format(k))
    # Weights that do not have match from the pre-trained model.
    not_use_layers = [
        k for k in pre_train_dict.keys() if k not in pre_train_dict_match.keys()
    ]
    # Log weights that are not loaded with the pre-trained weights.
    if not_use_layers:
        for k in not_use_layers:
            logger.info("Network weights {} not used.".format(k))
    # Load pre-trained weights.
    ms.load_state_dict(pre_train_dict_match, strict=False)
    epoch = -1

    # Load the optimizer state (commonly not done when fine-tuning)
    if "epoch" in checkpoint.keys() and not epoch_reset:
        epoch = checkpoint["epoch"]
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler_state"])
    else:
        epoch = -1
    return epoch


# instantiate model object
curr_model = MViT(cfg)
#  pretrained = torch.load('mvit/MViTv2_T_in1k.pyth')
# CS10-2: Minjun Jung #
"""
    The above pretrained variable wasn't even used anywhere else later,
    and this block is on __main__ execution indentation block.
"""
# End of Comment #
# my_checkpoint_path = "mvit/current_model_weights.pyth"

load_checkpoint(path_to_checkpoint=my_checkpoint_path, model=curr_model)
cfg.TRAIN.CHECKPOINT_FILE_PATH = my_checkpoint_path

if int(config_obj.get("CHECKPOINT","RESET_CHECKPOINT")) >= 1:
    print(f'loaded checkpoint')
    cu.save_checkpoint(
    cfg.OUTPUT_DIR,
    my_checkpoint_path,
    curr_model,
    optim.construct_optimizer(curr_model, cfg),
    0, # epoch to reset to 
    cfg,
    None
    )

# creating function to freeze weights - this is pointed at curr_model by default 
def freeze_weights(blocks_frozen):
    block_freeze = int(blocks_frozen) # freeze this many blocks starting from the back
    for i in range(0,len(curr_model.blocks)):
        if i >= block_freeze:
            for name, m in curr_model.blocks[i].named_modules():
                if 'norm' in name: # if norm layer, make sure it stays frozen
                    for p in m.parameters(recurse=False):
                        p.requires_grad = False 
                else:
                    for p in m.parameters(recurse=False):
                        p.requires_grad=True
        else:
            for name, m in curr_model.blocks[i].named_modules(): 
                if 'norm' in name: # if norm layer, make sure it stays frozen
                    for p in m.parameters(recurse=False):
                        p.requires_grad = False 
                else:
                    for p in m.parameters(recurse=False):
                        p.requires_grad = False # other unfreeze
    logging.info(f'First {blocks_frozen} blocks frozen out of {len(curr_model.blocks)}')


""" ***************
    Minjun Jung:
    Instead of pickling only the weights, which made it harder for us to load,
    maybe try using save_checkpoint() function already defined by MViT?

    There's a routine that calls save_checkpoint() already,
    and the only thing needed to activate seems to be a cfg. flag.
    which calculates True & False depending on the below variable's value.

    PROBLEM FIXED.
    Use this (↓) variable to change weights' saving period.
    When reloading the weight's replace above
        my_checkpoint_path = "mvit/MViTv2_B_in1k.pyth"
    with
        my_checkpoint_path = "tmp/checkpoints/[checkpoint_name].pyth"
    or just move the pyth file to main dir and rename accordingly.
"""
cfg.TRAIN.CHECKPOINT_PERIOD = 1  # save to current_weights.pyth every epoch.


DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.
The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""

def build_dataset(dataset_name, cfg, split):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    # Capitalize the the first letter of the dataset_name since the dataset_name
    # in configs may be in lowercase but the name of dataset class should always
    # start with an uppercase letter.
    name = dataset_name.capitalize()
    return DATASET_REGISTRY.get(name)(cfg, split)

def multiple_samples_collate(batch):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]

    inputs, labels = default_collate(inputs), default_collate(labels)

    return inputs, labels


def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

    if cfg.AUG.NUM_SAMPLE > 1 and split in ["train"]:
        collate_func = multiple_samples_collate
    else:
        collate_func = None

    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=collate_func,
    )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the dataset.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = loader.sampler
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)

# %%
#transform.py
import random

import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms

from mvit.datasets.rand_augment import rand_augment_transform # edited location
from mvit.datasets.random_erasing import RandomErasing # edited location

_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _pil_interp(method):
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        return Image.BILINEAR


# The following code are modified based on timm lib, we will replace the following
# contents with dependency from PyTorchVideo.
# https://github.com/facebookresearch/pytorchvideo
class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
    ):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            logging.info("range should be of kind (min, max)")

        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = " ".join(
                [_pil_interpolation_to_str[x] for x in self.interpolation]
            )
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += ", scale={0}".format(tuple(round(s, 4) for s in self.scale))
        format_string += ", ratio={0}".format(tuple(round(r, 4) for r in self.ratio))
        format_string += ", interpolation={0})".format(interpolate_str)
        return format_string


def transforms_imagenet_train(
    img_size=224,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    interpolation="random",
    use_prefetcher=False,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_num_splits=0,
    separate=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    if isinstance(img_size, tuple):
        img_size = img_size[-2:]
    else:
        img_size = img_size

    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
    primary_tfl = [
        RandomResizedCropAndInterpolation(
            img_size, scale=scale, ratio=ratio, interpolation=interpolation
        )
    ]
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith("augmix"):
            raise NotImplementedError("Augmix not implemented")
        else:
            raise NotImplementedError("Auto aug not implemented")
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    final_tfl += [
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    if re_prob > 0.0:
        final_tfl.append(
            RandomErasing(
                re_prob,
                mode=re_mode,
                max_count=re_count,
                num_splits=re_num_splits,
                device="cpu",
                cube=False,
            )
        )

    if separate:
        return (
            transforms.Compose(primary_tfl),
            transforms.Compose(secondary_tfl),
            transforms.Compose(final_tfl),
        )
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)

import json
import torch.utils.data
from mvit.utils.env import pathmgr
from torchvision import transforms as transforms_tv
from mvit.datasets.transform import transforms_imagenet_train

# MAYBE CAN DELETE IMAGENET class
@DATASET_REGISTRY.register()
class Imagenet(torch.utils.data.Dataset):
    """ImageNet dataset."""

    def __init__(self, cfg, mode, num_retries=10):
        self.num_retries = num_retries
        self.cfg = cfg
        self.mode = mode
        self.data_path = cfg.DATA.PATH_TO_DATA_DIR
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for ImageNet".format(mode)
        logger.info("Constructing ImageNet {}...".format(mode))
        if cfg.DATA.PATH_TO_PRELOAD_IMDB == "":
            self._construct_imdb()
        else:
            self._load_imdb()

    def _load_imdb(self):
        split_path = os.path.join(
            self.cfg.DATA.PATH_TO_PRELOAD_IMDB,
            f"{self.mode}.json" if self.mode != "test" else "val.json",
        )
        with pathmgr.open(split_path, "r") as f:
            data = f.read()
        self._imdb = json.loads(data)
        

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        split_path = os.path.join(self.data_path, self.mode)
        logger.info("{} data path: {}".format(self.mode, split_path))
        # Images are stored per class in subdirs (format: n<number>)
        split_files = pathmgr.ls(split_path)
        self._class_ids = sorted(f for f in split_files #if re.match(r"^n[0-9]+$", f) # LF: removing this regex because it's filtered out our files 
        )
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Construct the image db
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)
            for im_name in pathmgr.ls(im_dir):
                im_path = os.path.join(im_dir, im_name)
                self._imdb.append({"im_path": im_path, "class": cont_id})
        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))

    def _prepare_im(self, im_path):
        with pathmgr.open(im_path, "rb") as f:
            with Image.open(f) as im:
                im = im.convert("RGB")
        # Convert HWC/BGR/int to HWC/RGB/float format for applying transforms
        train_size, test_size = (
            self.cfg.DATA.TRAIN_CROP_SIZE,
            self.cfg.DATA.TEST_CROP_SIZE,
        )

        if self.mode == "train":
            aug_transform = transforms_imagenet_train(
                img_size=(train_size, train_size),
                color_jitter=self.cfg.AUG.COLOR_JITTER,
                auto_augment=self.cfg.AUG.AA_TYPE,
                interpolation=self.cfg.AUG.INTERPOLATION,
                re_prob=self.cfg.AUG.RE_PROB,
                re_mode=self.cfg.AUG.RE_MODE,
                re_count=self.cfg.AUG.RE_COUNT,
                mean=self.cfg.DATA.MEAN,
                std=self.cfg.DATA.STD,
            )
        else:
            t = []
            if self.cfg.DATA.VAL_CROP_RATIO == 0.0:
                t.append(
                    transforms_tv.Resize((test_size, test_size), interpolation=3),
                )
            else:
                # size = int((256 / 224) * test_size) # = 1/0.875 * test_size
                size = int((1.0 / self.cfg.DATA.VAL_CROP_RATIO) * test_size)
                t.append(
                    transforms_tv.Resize(
                        size, interpolation=3
                    ),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms_tv.CenterCrop(test_size))
            t.append(transforms_tv.ToTensor())
            t.append(transforms_tv.Normalize(self.cfg.DATA.MEAN, self.cfg.DATA.STD))
            aug_transform = transforms_tv.Compose(t)
        im = aug_transform(im)
        return im

    def __load__(self, index):
        try:
            # Load the image
            im_path = self._imdb[index]["im_path"]
            # Prepare the image for training / testing
            if self.mode == "train" and self.cfg.AUG.NUM_SAMPLE > 1:
                im = []
                for _ in range(self.cfg.AUG.NUM_SAMPLE):
                    crop = self._prepare_im(im_path)
                    im.append(crop)
                return im
            else:
                im = self._prepare_im(im_path)
                return im

        except Exception as e:
            logging.error(e)
            return None

    def __getitem__(self, index):
        # if the current image is corrupted, load a different image.
        for _ in range(self.num_retries):
            im = self.__load__(index)
            # Data corrupted, retry with a different image.
            if im is None:
                index = random.randint(0, len(self._imdb) - 1)
            else:
                break
        # Retrieve the label
        label = self._imdb[index]["class"]
        if isinstance(im, list):
            label = [label for _ in range(len(im))]

        return im, label

    def __len__(self):
        return len(self._imdb)


# MAYBE CAN DELETE: imagenettrain
imagenettrain = Imagenet(cfg,"train")
train_loader = construct_loader(cfg, "train")
val_loader = construct_loader(cfg, "val")

# think you need to use this function to convert labels if you turn off the mixup function
def convert_to_one_hot(targets, num_classes, on_value=1.0, off_value=0.0):
    """
    This function converts target class indices to one-hot vectors, given the
    number of classes.
    Args:
        targets (loader): Class labels.
        num_classes (int): Total number of classes.
        on_value (float): Target Value for ground truth class.
        off_value (float): Target Value for other classes.This value is used for
            label smoothing.
    """

    targets = targets.long().view(-1, 1)
    return torch.full(
        (targets.size()[0], num_classes), off_value, device=targets.device
    ).scatter_(1, targets, on_value)

# %%
### creating these lists to debug inputs in train_epoch function
# holder_list = []
# holder_list2= []
# holder_list3= []

# making our own function for errors
def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.
    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.
    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    _toplabelval, labels = torch.topk(
        labels, max([1]), dim=1, largest=True, sorted=True
    )

    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct

# %%
# making our own function for errors
def topks_correcttest(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.
    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.
    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    _toplabelval, labels = torch.topk(
        labels, max([1]), dim=1, largest=True, sorted=True
    )

    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct




def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
):
    """
    Perform the training for one epoch.
    Args:
        train_loader (loader): training loader.
        model (model): the model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        scaler (GradScaler): the `GradScaler` to help perform the steps of gradient scaling.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            mvit/config/defaults.py
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    
    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )
        
    # pred_holder = torch.empty(size=[0,cfg.MODEL.NUM_CLASSES])
    # label_holder = torch.empty(size=[0,cfg.MODEL.NUM_CLASSES])
    # clean_label_holder = torch.empty(size=[0],dtype=torch.int8)
    f1_holder = []
    top1_err_holder=[]
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        logging.info(f'Epoch {cur_epoch}: cur_iter {cur_iter}')
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
        if use_mac==1:
            inputs = inputs.to(device)
            labels = labels.to(device)
        # lf - turning this off - stil need to one hot encode labels though
        if cfg.MIXUP.ENABLE:
            inputs, labels = mixup_fn(inputs, labels)
        # LF: adding this so that the labels are format [4,1,1000]. Existing form is just a tensor of dimension 4 whwen MIXUP.ENABLE is turned off
        labels = torch.unsqueeze(convert_to_one_hot(labels,cfg.MODEL.NUM_CLASSES),dim=1) # use this one if the preds are shape [batch_size,49,1000]
        #labels = convert_to_one_hot(labels,cfg.MODEL.NUM_CLASSES) # use this if the preds are shape [batch_size,1000]
        # holder_list2.append([inputs,labels]) # testing only
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)
        logging.info(f'Current learning rate: {lr:.5f}')
        #train_meter.data_toc()
        
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            preds = model(inputs)
            if use_mac==1:
                preds.to(device)
            # holder_list3.append([inputs,labels,preds])
            ######################### ignore 
            ## LF: inserting own function to show predictions vs labels
            # testpred_outcome_topk = torch.topk(preds,k=5,dim=1).indices
            # testlabel_outcome = torch.argmax(labels,dim=1,keepdim=True)
            # for i in range(0,len(testlabel_outcome)):
            #     logging.info(f'P: {testpred_outcome_topk[i]}, A: {testlabel_outcome[i]}')
            ###########################
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
            # Compute the loss
            loss = loss_fun(preds, labels)
            
            
        # check Nan Loss.
        misc.check_nan_losses(loss)
        # Perform the backward pass.
        #logging.info('running backward pass')
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        # if cfg.SOLVER.CLIP_GRAD_VAL:
        #     torch.nn.utils.clip_grad_value_(
        #         model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
        #     )
        # elif cfg.SOLVER.CLIP_GRAD_L2NORM:
        #     torch.nn.utils.clip_grad_norm_(
        #         model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
        #     )
        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()


#         if cfg.MIXUP.ENABLE:
#             _top_max_k_vals, top_max_k_inds = torch.topk(
#                 labels, 2, dim=1, largest=True, sorted=True
#             )
#             idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
#             idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
#             preds = preds.detach()
#             preds[idx_top1] += preds[idx_top2]
#             preds[idx_top2] = 0.0
#             labels = top_max_k_inds[:, 0]

        # LF: change format of preds and labels before passing into error functions
        preds = torch.squeeze(torch.mean(preds,dim=1,keepdim=True),dim=1)
        labels = torch.squeeze(torch.mean(labels,dim=1,keepdim=True),dim=1)

# =============== EDITED BY CS10-2: Minjun Jung BEGIN *************** #
        # if torch.cuda.is_available():
        #     pred_holder = torch.cat((pred_holder.to(torch.device("cuda")),preds),dim=0)
        #     label_holder = torch.cat((label_holder.to(torch.device("cuda")),labels),dim=0)
        # elif use_mac==1:
        #     pred_holder = torch.cat((pred_holder.to(device),preds),dim=0)
        #     label_holder = torch.cat((label_holder.to(device),labels),dim=0)
        # else:
        #     pred_holder = torch.cat((pred_holder,preds),dim=0)
        #     label_holder = torch.cat((label_holder,labels),dim=0)
        # Tensor types have to be synchronized to CUDA types.
# =============== EDITED BY CS10-2: Minjun Jung END =============== #
        
        #num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        num_topks_correct = topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])
        #if torch.has_mps:
        #    loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top5_err = (
            loss.item(),
            top1_err.item(),
            top5_err.item(),
        )
        
        #clean_preds = torch.argmax(preds,dim=1)
        clean_labels = torch.argmax(labels,dim=1)
        
# =============== EDITED BY CS10-2: Minjun Jung BEGIN *************** #
        # if torch.cuda.is_available():
        #     clean_label_holder = torch.cat((clean_label_holder.type(torch.LongTensor)
        #                                     .to(torch.device("cuda")),clean_labels),dim=0)
        # elif use_mac==1:
        #     clean_label_holder = torch.cat((clean_label_holder.type(torch.LongTensor)
        #                                     .to(device),clean_labels),dim=0)
        # else:
        #     clean_label_holder = torch.cat((clean_label_holder.type(torch.LongTensor),clean_labels),dim=0)

        # Tensor types have to be synchronized to CUDA types.
        # {CharTensor} -> {LongTensor} -> .to(cuda)
# =============== EDITED BY CS10-2: Minjun Jung END =============== #
        

        # LF: calculate f1 score from torchmetrics - maybe we save this one for later
# =============== EDITED BY CS10-2: Minjun Jung BEGIN *************** #
        if torch.cuda.is_available():
            f1func = F1Score(num_classes=cfg.MODEL.NUM_CLASSES).to(torch.device("cuda"))
        elif use_mac==1:
            f1func = F1Score(num_classes=cfg.MODEL.NUM_CLASSES).to(device)
        else:
            f1func = F1Score(num_classes=cfg.MODEL.NUM_CLASSES)
        # Even performance metric functions have to sync the device.
# =============== EDITED BY CS10-2: Minjun Jung END =============== #
        
        batch_f1 = f1func(preds, clean_labels)
        top1_err_holder.append(top1_err)
        f1_holder.append(batch_f1.cpu())
        # prediction printing lf
        #logging.info(f'Predicting {clean_preds}')
        # label
        #logging.info(f'Labels... {clean_labels}')
        
        # only print every X iterations
        if cur_iter % 5 == 0:
            logging.info(f'Current top1_err: {top1_err:.3f}, Current top5_err: {top5_err:.3f}')
            logging.info(f'Current f1: {batch_f1:.3f}')
            logging.info(f'Loss {loss:.3f}\n')

    top1_err = np.mean(top1_err_holder)
    f1 = np.mean(f1_holder)
    return f1, top1_err # return final f1

def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

    if cfg.AUG.NUM_SAMPLE > 1 and split in ["train"]:
        collate_func = multiple_samples_collate
    else:
        collate_func = None

    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=collate_func,
    )
    return loader


def train(cfg):
    """
    Train a model on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in mvit/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # just loading the model not using registry
    model = curr_model
    freeze_weights(my_layers_frozen) # freeze weights as specified into config.ini file

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Create the train and val loaders.
    train_loader = construct_loader(cfg, "train")
    val_loader = construct_loader(cfg, "val")

    logging.info(f'LENGTH OF TRAIN_LOADER {len(train_loader)}')

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # eval f1 holder
    eval_f1_holder = []


    # =============== Added by Minjun Jung ***************
    dom_epoch_wise = {"epoch": [], "prec": [], "rec": [], "f1": []}
    # Due to so many similar-named objects with similar role
    # I made this unique name not to confuse with other performance results.
    # =============== Added by Minjun Jung ===============


    start_epoch = load_checkpoint(path_to_checkpoint=my_checkpoint_path, model=curr_model)

    epoch_timer = EpochTimer()
    logging.info(f'Going from start_epoch: {start_epoch+1} to.. {cfg.SOLVER.MAX_EPOCH}\n\n')
    for cur_epoch in range(start_epoch+1, cfg.SOLVER.MAX_EPOCH):
        logging.info(f'**Epoch no {cur_epoch}')
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_f1,train_err = train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
        )
        epoch_timer.epoch_toc()
        logger.info(
           f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
           f"from {start_epoch} to {cur_epoch} take "
           f"{epoch_timer.avg_epoch_time():.2f}s in average and "
           f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
           f"For epoch {cur_epoch}, each iteraction takes "
           f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
           f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
           f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
        )
        is_eval_epoch = misc.is_eval_epoch(cfg, cur_epoch)

        # Save a checkpoint.
        """
        cu.save_checkpoint(
            cfg.OUTPUT_DIR,
            my_checkpoint_path,
            model,
            optimizer,
            cur_epoch,
            cfg,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
        )
        """
        
        
        # Evaluate the model on validation set.
        # coding up so that we can compare train and validation f1 
        fin_results['epoch'].append(cur_epoch)
        fin_results['train_f1'].append(train_f1)
        fin_results['train_err'].append(train_err)
        if is_eval_epoch:
            eval_f1,eval_err = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)
            # compare difference in f1
            fin_results['validation_f1'].append(eval_f1)
            fin_results['validation_err'].append(eval_err)
            eval_f1_holder.append(eval_f1)
            if len(eval_f1_holder) >= 2:
                diff = eval_f1_holder[-1] - eval_f1_holder[-2]
                logging.info(f'Current epoch F1 is {eval_f1} vs prior F1 {eval_f1_holder[-2]} Difference in epoch is {diff}')
        else:
            fin_results['validation_f1'].append(np.NaN)
            fin_results['validation_err'].append(np.NaN)
        pd_fin_results = pd.DataFrame(fin_results)
        pd_fin_results.to_csv(new_dir + "/" + datetime.now().strftime("%Y_%m_%d_%H%M%S") +\
            "_tracking_results.csv")  # Put the files into their correct time-stamped folder.
        
        
        # =============== Added by Minjun Jung ***************
        """
            The previous way of calculating mean F1 can be misleading if the batch size is too small.
            Hence to get a better idea how epoch-wise F1 performance is doing we decided to use my
            stats function to calculate over the entire val dataset.
        """

        # Get predictions on the whole validation data for stat calculation.
        y_true, y_pred = predict_val(val_loader, model, cfg)
        val_labels = {v: k for k, v in val_loader.dataset._class_id_cont_id.items()}
        acc, prec, rec, f1 = getPerformance(y_true, y_pred, val_labels,
            title="Epoch " + str(cur_epoch) + " Confusion Matrix", dir=new_dir)


        # Minjun: moved check point saving into if, so that weights are saved only if
        #    the performance improved than the previous best F1.
        #    Weights are saved as "Epoch_XX.pyth" and the number of XX (epochs)
        #    should be increasing if observed in the elbow plot.
        if len(dom_epoch_wise["f1"]) == 0 or np.nanmean(f1) > max(dom_epoch_wise["f1"]):
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                my_checkpoint_path,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )


        # Save the mean results to epoch-wise result container.
        dom_epoch_wise["epoch"].append(cur_epoch)
        dom_epoch_wise["prec"].append(np.nanmean(prec))
        dom_epoch_wise["rec"].append(np.nanmean(rec))
        dom_epoch_wise["f1"].append(np.nanmean(f1))
        # =============== Added by Minjun Jung ===============

        # KN: plot elbow chart every x_epoch epochs
        x_epoch = 1
        if ((cur_epoch+1) % x_epoch == 0) or ((cur_epoch+1)==cfg.SOLVER.MAX_EPOCH):
            logging.info(f"Exporting metrics convergence plot at {cur_epoch}")
            plotElbow(dom_epoch_wise, title="metrics convergence elbow chart", dir=new_dir)
    
    return dom_epoch_wise


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            mvit/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    # pred_holder = torch.empty(size=[0,cfg.MODEL.NUM_CLASSES])
    # label_holder = torch.empty(size=[0,cfg.MODEL.NUM_CLASSES])
    # clean_label_holder = torch.empty(size=[0],dtype=torch.int8)
    # error_holder=torch.empty(size=[0,cfg.MODEL.NUM_CLASSES])
    f1_holder = []
    top1_err_holder = []
    for cur_iter, (inputs, labels) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
        if use_mac==1:
            # Transferthe data to the current GPU device.
            inputs = inputs.to(device)
            labels = labels.to(device)

        #val_meter.data_toc()
        preds = model(inputs)
        if use_mac==1:
            preds.to(device)
        labels = torch.unsqueeze(convert_to_one_hot(labels,cfg.MODEL.NUM_CLASSES),dim=1)
        preds = torch.squeeze(torch.mean(preds,dim=1,keepdim=True),dim=1)
        labels = torch.squeeze(torch.mean(labels,dim=1,keepdim=True),dim=1)

# =============== EDITED BY CS10-2: Minjun Jung BEGIN *************** #
        # if torch.cuda.is_available():
        #     pred_holder = torch.cat((pred_holder.to(torch.device("cuda")),preds),dim=0)
        #     label_holder = torch.cat((label_holder.to(torch.device("cuda")),labels),dim=0)
        # elif use_mac==1:
        #     pred_holder = torch.cat((pred_holder.to(device),preds),dim=0)
        #     label_holder = torch.cat((label_holder.to(device),labels),dim=0)
        # else:
        #     pred_holder = torch.cat((pred_holder,preds),dim=0)
        #     label_holder = torch.cat((label_holder,labels),dim=0)
        # Tensor types have to be synchronized to CUDA types.
# =============== EDITED BY CS10-2: Minjun Jung END =============== #
        
        #num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        num_topks_correct = topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.all_reduce([top1_err, top5_err])

        # Copy the stats from GPU to CPU (sync point).
        top1_err, top5_err = (
            top1_err.item(),
            top5_err.item(),
        )
        

        clean_labels = torch.argmax(labels,dim=1)
        
# =============== EDITED BY CS10-2: Minjun Jung BEGIN *************** #
        # if torch.cuda.is_available():
        #     clean_label_holder = torch.cat((clean_label_holder.type(torch.LongTensor)
        #                                     .to(torch.device("cuda")),clean_labels),dim=0)
        # elif use_mac==1:
        #     clean_label_holder = torch.cat((clean_label_holder.type(torch.LongTensor)
        #                                     .to(device),clean_labels),dim=0)                              
        # else:
        #     clean_label_holder = torch.cat((clean_label_holder.type(torch.LongTensor),clean_labels),dim=0)

        # Tensor types have to be synchronized to CUDA types.
        # {CharTensor} -> {LongTensor} -> .to(cuda)
# =============== EDITED BY CS10-2: Minjun Jung END =============== #


        # LF: calculate f1 score from torchmetrics - maybe we save this one for later
# =============== EDITED BY CS10-2: Minjun Jung BEGIN *************** #
        if torch.cuda.is_available():
            f1func = F1Score(num_classes=cfg.MODEL.NUM_CLASSES).to(torch.device("cuda"))
        elif use_mac==1:
            f1func = F1Score(num_classes=cfg.MODEL.NUM_CLASSES).to(device)
        else:
            f1func = F1Score(num_classes=cfg.MODEL.NUM_CLASSES)
        # Even performance metric functions have to sync the device.
# =============== EDITED BY CS10-2: Minjun Jung END =============== #
        
        f1_batch = f1func(preds, clean_labels)
        f1_holder.append(f1_batch.cpu())
        top1_err_holder.append(top1_err)

        # prediction printing lf
        #logging.info(f'Predicting... {clean_preds}')
        # label
        #logging.info('Labels... {clean_labels}',)
        # only print every X iterations
        # logging.info(f'Batch eval_error: {top1_err:.3f}')
        # logging.info(f'Batch eval_F1: {f1_batch:.3f}')
    f1 = np.mean(f1_holder)
    top1_err = np.mean(top1_err_holder)
    logging.info('\n\n')
    logging.info(f'Final eval_error: {top1_err:.3f}')
    logging.info(f'Final eval_F1: {f1:.3f}')
    return f1, top1_err


# seems like loading the pre-trained model leads to issues?
# what if we try
# if you turn off scaler.step(optimizer) it seems to work

# creating results dictionary to capture trainning results
fin_results = {'epoch':[], 'train_f1':[], 'validation_f1':[],'train_err':[],'validation_err':[]}


# Minjun Jung: ***************
# In order to use label-wise val prediction & stat use below function.

def report_on_valid(val_loader, curr_model, cfg):
    y_true, y_pred = predict_val(val_loader, curr_model, cfg)
    val_labels = {v: k for k, v in val_loader.dataset._class_id_cont_id.items()}
    getPerformance(y_true, y_pred, val_labels, title="Confusion Matrix", dir=new_dir)
# This can still be used as its own independent function to run on validation data
# without having to run train(), but the epoch-wise F1 needed some modification from this
# so I decided to copy required parts into train() function.
# ===============



if __name__ == '__main__':
    # Minjun Jung: Create folder with unique timestamps everytime the training happens.
    new_dir = "./logs/" + datetime.now().strftime("%Y_%m_%d_%H%M%S")
    os.mkdir(new_dir)
    shutil.copyfile(src=config_path, dst=new_dir + "/" + config_path.split("/")[-1])
    setupLogging(new_dir)

    start = time.time()
    logging.info(f"Start time: {start}")
    epoch_results = train(cfg)
    printEpochWise(epoch_results)
    logging.info(f"Running evaluation report...")
    # report_on_valid(val_loader, curr_model, cfg)
    # Minjun Jung: above used to print report only once at the end.
    finish = time.time()
    logging.info("***** " + str(round(finish - start, 3)) + " secs *****")


################### LAYER FREEZING ##################################
# test weight freezing - this just needs to not change after certain number of iterations
#curr_model.state_dict()['patch_embed.proj.weight']

# UNFREEZE WEIGHTS
# layer_counter = 0
# for name, m in curr_model.named_modules():
#     for p in m.parameters(recurse=False):
#         # if layer_counter > x ...
#         p.requires_grad = True
#     layer_counter += 1


# # FREEZE WEIGHTS after layer count X 
# count=1
# for param in curr_model.parameters():
#     count += 1
#     if count >= 5: # this needs to be greater than or equal to 1 because at least one parameter needs to be unfrozen
#         param.requires_grad = False


# # check what's frozen - navigate to just the blocks
# for name, m in curr_model.named_modules():
#     for p in m.parameters(recurse=False):
#         if p.requires_grad == True:
#             print(name,'Unfrozen')
#         else:
#             print(name,'Frozen')

#####################################################            


### LF checks that load_checkpoint actually works 
# LF checking current model weights before any training
# curr_model.state_dict()

# just check first layer - yep looks the same at the tail 
# pretrained['model_state']['patch_embed.proj.weight']
# curr_model.state_dict()['patch_embed.proj.weight']

# # now after training - check the output has changed .. 
# curr_model.state_dict()['patch_embed.proj.weight']

# save current model weights
# cu.save_checkpoint(
# './',
# curr_model,
# optim.construct_optimizer(curr_model, cfg),
# 0,
# cfg,
# None
# )



# check that there's something saved in the drive..

# now let's try to refresh the model 
#curr_model = MViT(cfg)
# refreshed model weights - nonsensical yep
#curr_model.state_dict()['patch_embed.proj.weight']

# now load the previous model
#load_checkpoint(path_to_checkpoint='./checkpoints/checkpoint_epoch_00001.pyth', model=curr_model)

# now check whether it's updated correctly - yes it has 
#curr_model.state_dict()['patch_embed.proj.weight']


