# utils.py
import os, random
import numpy as np

# ----- constants -----
FREE, OBST, JUNK = 0, 1, 2
GRID_SIZE = 64
TARGET_FILL = 0.20
NUM_ENEMIES = 10
NEI_ORDER = [(-1,0), (0,1), (1,0), (0,-1)]
SEED = 42
SHOW_FIGURES = True   # set True for windows mode, False for report mode


rng = np.random.default_rng(SEED)
random.seed(SEED)

# ----- helpers -----
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def sign(x: int) -> int:
    return (x > 0) - (x < 0)

