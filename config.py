# config.py
import torch
from pathlib import Path

# 网格 & 物理
NX, NY = 100, 100
RESOLUTION = 20
DPML_PX = 12

# 时序
T_STEPS = 32
T_WARMUP = 200
SAMPLE_EVERY_MIN = 1
SAMPLE_EVERY_MAX = 3

# 源频率/振幅/带宽
F_MIN, F_MAX = 0.05, 0.55
FWIDTH_MIN_RATIO = 0.02
FWIDTH_MAX_RATIO = 0.35
AMP_MIN, AMP_MAX = 0.02, 2.00

# 采集
N_EPISODES = 160
MAX_SAMPLES_PER_EP = 1200
CLIP_VALUE = 20.0

BASE = Path("./runs_crbc")
DATA_DIR = BASE / "data"
CKPT_DIR = BASE / "checkpoints"
LOG_FILE = BASE / "train.log"
DATASET_NAME = "tmz_crbc_ds"

USE_MPI = False
SAVE_RAW_FRAMES = False
RNG_SEED = 2025

# 特征维度
PATCH_RADIUS = 2
D_INPUT = 38
D_MODEL = 192
D_STATE = 96
N_LAYERS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 训练
MAX_EPOCHS = 100
BATCH_SIZE = 512
LR = 1e-4
WEIGHT_DECAY = 1e-4
CLIP_NORM = 5.0
VAL_SPLIT = 0.2       
AMP = False             
EARLY_STOP_PATIENCE = 20

LAMBDA_EDGE = 1.0
LAMBDA_CORNER = 1.5

INPUT_NOISE_STD = 0.01  
