import sys
import logging
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Hyperparameters(BaseSettings):
    LR_RANGES: List[float] = [0.1, 0.05, 0.01, 0.001]
    L2_RANGES: List[float] = [0.0, 1e-6, 1e-5]
    
    HT_GRACE_PERIODS: List[int] = [100, 500, 1000]
    HT_SPLIT_CONFIDENCES: List[float] = [1e-5, 1e-7]
    
    FM_FACTORS: List[int] = [8, 16]
    FM_WEIGHT_DECAY: List[float] = [0.001, 0.0001]
    
    NN_LAYERS: List[List[int]] = [
        [32, 16],
        [64, 32, 16],
    ]
    NN_LR: List[float] = [0.01, 0.005]

class SystemConfig(BaseSettings):
    APP_NAME: str = "RTB Dashboard"
    ENV: str = Field(default="dev")

    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    LOG_DIR: Path = BASE_DIR / "logs"
    ASSET_DIR: Path = BASE_DIR / "assets"
    
    DATA_SOURCE_MODE: str = "manual"
    DATASET_FILENAME: str = "train.txt"
    MAX_EVENTS: int = 50000        
    DRIFT_POINT: int = 25000       
    DRIFT_TYPE: str = "feature_swap"
    METRICS_WINDOW_SIZE: int = 1000
    LOG_INTERVAL: int = 100
    CONSOLE_PRINT_INTERVAL: int = 2000

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = SystemConfig()
params = Hyperparameters()
for path in [settings.RAW_DIR, settings.LOG_DIR, settings.ASSET_DIR]:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"CRITICAL: Failed to initialize directory {path}. Error: {e}")
        sys.exit(1)