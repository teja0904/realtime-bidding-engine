import sys
import argparse
import logging
import time
import subprocess
import traceback
from typing import List
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

try:
    from config import settings, params
    from data_stream import StreamGenerator
    from evaluator import PrequentialEvaluator
    
    from models import (
        OnlineModel,
        RiverLogisticRegression,
        RiverHoeffdingTree,
        RiverFM,
        StreamingNeuralNet
    )
    
    import dashboard 
    
except ImportError as e:
    print("\nError: Missing dependencies.")
    print(f"Details: {e}")
    print("Action: Please run 'pip install -r requirements.txt'\n")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_DIR / "system.log", mode='w'), # Overwrite log on new run
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Main")

def build_competitive_grid() -> List[OnlineModel]:
    logger.info("Building model grid...")
    models = []

    for lr in params.LR_RANGES:
        for l2 in params.L2_RANGES:
            name = f"LogReg_lr{lr}_l2{l2}"
            models.append(RiverLogisticRegression(name, lr=lr, l2=l2))

    for grace in params.HT_GRACE_PERIODS:
        for conf in params.HT_SPLIT_CONFIDENCES:
            name = f"HAT_grace{grace}_conf{conf}"
            models.append(RiverHoeffdingTree(name, grace_period=grace, split_confidence=conf))

    for factors in params.FM_FACTORS:
        for decay in params.FM_WEIGHT_DECAY:
            name = f"FM_dim{factors}_decay{decay}"
            models.append(RiverFM(name, n_factors=factors, weight_decay=decay))

    for layers in params.NN_LAYERS:
        for lr in params.NN_LR:
            layer_str = "-".join(map(str, layers))
            name = f"NN_{layer_str}_lr{lr}"
            models.append(StreamingNeuralNet(name, input_dim=40, hidden_layers=layers, lr=lr))

    logger.info(f"Grid populated: {len(models)} models.")
    return models

def run_pipeline():
    logger.info(f"Starting {settings.APP_NAME} pipeline")
    logger.info(f"Mode: {settings.DATA_SOURCE_MODE} | Drift: {settings.DRIFT_TYPE} @ {settings.DRIFT_POINT}")

    logger.info("Step 1: Connecting to data stream...")
    try:
        stream_factory = StreamGenerator()
        stream = stream_factory.stream()
    except Exception as e:
        logger.critical(f"Stream Connection Failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    logger.info("Step 2: Initializing models...")
    try:
        models = build_competitive_grid()
        if not models:
            logger.error("No models found! Check config.Hyperparameters.")
            sys.exit(1)
    except Exception as e:
        logger.critical(f"Model Build Failed: {e}")
        sys.exit(1)

    logger.info("Step 3: Starting prequential evaluation...")
    
    evaluator = PrequentialEvaluator(models)
    start_time = time.time()
    
    try:
        evaluator.evaluate(stream)
        
        duration = time.time() - start_time
        logger.info(f"Benchmark done in {duration:.2f}s ({settings.MAX_EVENTS / duration:.0f} events/sec)")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
    except Exception as e:
        logger.critical(f"Runtime Error during Evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)

    logger.info("Step 4: Generating assets...")
    try:
        output_path = dashboard.generate_architecture_diagram(save_to_disk=True)
        logger.info(f"Architecture saved to: {output_path}")
    except Exception as e:
        logger.error(f"Asset Generation Failed: {e}")

    logger.info("Run complete. Use 'python main.py ui' to view results.")

def run_ui():
    logger.info("Launching dashboard...")
    dashboard_path = PROJECT_ROOT / "dashboard.py"
    
    if not dashboard_path.exists():
        logger.critical(f"Missing dashboard.py at {dashboard_path}")
        sys.exit(1)

    cmd = ["streamlit", "run", str(dashboard_path)]
    
    try:
        # Use shell=False for security, check=True to catch crashes
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Streamlit crashed with code {e.returncode}.")
    except FileNotFoundError:
        logger.error("'streamlit' command not found. Install it: pip install streamlit")
    except KeyboardInterrupt:
        logger.info("Dashboard stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTB Pipeline CLI")
    parser.add_argument(
        "command",
        choices=["pipeline", "ui"],
        help="pipeline: Run the training benchmark | ui: Launch the visual dashboard"
    )
    
    # Print help if no args provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    if args.command == "pipeline":
        run_pipeline()
    elif args.command == "ui":
        run_ui()