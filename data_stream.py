import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Dict, Any, Tuple, Optional

try:
    from config import settings
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from config import settings

logger = logging.getLogger("StreamFactory")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)

@dataclass
class StreamRecord:
    label: int
    features: Dict[str, Any]

class StreamSchema:
    COLUMNS = ["Label"] + [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
    
    @classmethod
    def normalize_row(cls, row: pd.Series) -> Dict[str, Any]:
        x = {}
        
        for col in cls.COLUMNS[1:14]:
            val = row.get(col)
            if pd.isna(val) or val == "":
                x[col] = 0
            else:
                try:
                    x[col] = int(val)
                except (ValueError, TypeError):
                    x[col] = 0
            
        for col in cls.COLUMNS[14:]:
            val = row.get(col)
            if pd.isna(val) or val == "":
                x[col] = "missing"
            else:
                x[col] = str(val)
                
        return x

class DriftInjector:
    def __init__(self, drift_type: str, drift_point: int):
        self.drift_type = drift_type
        self.drift_point = drift_point
        self._active = False

    def apply(self, x: Dict[str, Any], y: int, count: int) -> Tuple[Dict[str, Any], int]:
        if count < self.drift_point:
            return x, y

        if not self._active:
            logger.warning(f"Drift injected: {self.drift_type} at event {count}")
            self._active = True

        x_corrupted = x.copy()
        
        if self.drift_type == "feature_swap":
            if 'I1' in x_corrupted and 'I5' in x_corrupted:
                x_corrupted['I1'], x_corrupted['I5'] = x_corrupted['I5'], x_corrupted['I1']

        elif self.drift_type == "label_flip":
            return x, 1 - y

        return x_corrupted, y

class StreamGenerator:
    def __init__(self):
        self.raw_path = settings.RAW_DIR / settings.DATASET_FILENAME
        self._validate_file()
        self.drifter = DriftInjector(settings.DRIFT_TYPE, settings.DRIFT_POINT)

    def _validate_file(self):
        if not self.raw_path.exists():
            logger.critical(f"Dataset missing: {self.raw_path}")
            logger.critical("Download the dataset manually. See README for instructions.")
            logger.critical("Please confirm 'train.txt' is inside the 'data/raw/' folder.")
            sys.exit(1)
        
        if self.raw_path.stat().st_size == 0:
            logger.critical(f"File empty: {self.raw_path} has 0 bytes.")
            sys.exit(1)
            
        size_mb = self.raw_path.stat().st_size / (1024 * 1024)
        logger.info(f"Dataset Verified: {self.raw_path.name} ({size_mb:.2f} MB)")

    def stream(self) -> Iterator[Tuple[Dict, int]]:
        logger.info("Initializing TSV reader...")
        
        try:
            reader = pd.read_csv(
                self.raw_path, 
                sep='\t', 
                header=None,
                names=StreamSchema.COLUMNS,
                chunksize=5000, 
                engine='c'
            )
        except Exception as e:
            logger.critical(f"Failed to initialize CSV Reader: {e}")
            sys.exit(1)
        
        counter = 0
        
        for chunk in reader:
            chunk = chunk.sample(frac=1.0)
            
            for _, row in chunk.iterrows():
                counter += 1
                
                try:
                    y = int(row['Label'])
                    x = StreamSchema.normalize_row(row)
                except Exception:
                    continue 

                x_final, y_final = self.drifter.apply(x, y, counter)
                yield x_final, y_final
                if counter >= settings.MAX_EVENTS:
                    logger.info(f"Simulation Target Reached ({settings.MAX_EVENTS} Events). Shutting down.")
                    return

if __name__ == "__main__":
    # Integration Test
    print("--- Stream Factory Integration Test ---")
    gen = StreamGenerator()
    stream = gen.stream()
    
    for i, (feat, label) in enumerate(stream):
        if i >= 5: break
        print(f"[{i}] Label: {label} | I1: {feat.get('I1')} | C1: {feat.get('C1')}")