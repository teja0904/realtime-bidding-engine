import time
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator, Tuple

import pandas as pd
from river import metrics

try:
    from config import settings
    from models import OnlineModel
except ImportError:
    pass

logger = logging.getLogger("Evaluator")

class MetricStore:
    def __init__(self):
        self.auc = metrics.ROCAUC()
        self.accuracy = metrics.Accuracy()
        self.log_loss = metrics.LogLoss()
        
        # Latency tracking (Rolling average of last N samples)
        self._latencies = [] 
        self._window = 1000

    def update(self, y_true: int, y_pred_proba: float, latency_ns: int):
        y_pred_label = 1 if y_pred_proba > 0.5 else 0
        
        self.auc.update(y_true, y_pred_proba)
        self.accuracy.update(y_true, y_pred_label)
        self.log_loss.update(y_true, y_pred_proba)
        
        # Update latency window
        self._latencies.append(latency_ns)
        if len(self._latencies) > self._window:
            self._latencies.pop(0)

    def get_snapshot(self) -> Dict[str, float]:
        avg_latency_us = (sum(self._latencies) / len(self._latencies)) / 1000.0 if self._latencies else 0.0
        return {
            "ROC-AUC": self.auc.get(),
            "Accuracy": self.accuracy.get(),
            "LogLoss": self.log_loss.get(),
            "Latency_us": avg_latency_us
        }

class PrequentialEvaluator:
    def __init__(self, models: List[OnlineModel]):
        self.models = models
        self.metric_stores = {model.name: MetricStore() for model in models}
        self.log_path = settings.LOG_DIR / "stream_metrics.csv"
        
        # Initialize CSV Header
        self._init_log_file()

    def _init_log_file(self):
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", 
                "event_id", 
                "model_name", 
                "auc", 
                "accuracy", 
                "logloss", 
                "latency_us"
            ])

    def _flush_logs(self, event_id: int, buffer: List[Dict]):
        if not buffer:
            return
            
        timestamp = pd.Timestamp.now().isoformat()
        
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            for record in buffer:
                writer.writerow([
                    timestamp,
                    event_id,
                    record['model'],
                    f"{record['auc']:.5f}",
                    f"{record['acc']:.5f}",
                    f"{record['loss']:.5f}",
                    f"{record['lat']:.1f}"
                ])

    def evaluate(self, stream_generator: Generator[Tuple[Dict, int], None, None]):
        logger.info(f"Starting Prequential Evaluation for {len(self.models)} models.")
        logger.info(f"Logging to: {self.log_path}")
        
        log_buffer = []
        start_time = time.time()
        
        for i, (x, y) in enumerate(stream_generator):
            
            for model in self.models:
                t0 = time.perf_counter_ns()
                try:
                    pred_proba = model.predict_one(x)
                except Exception as e:
                    logger.error(f"Model {model.name} failed prediction: {e}")
                    pred_proba = 0.5
                t1 = time.perf_counter_ns()
                
                latency = t1 - t0
                store = self.metric_stores[model.name]
                store.update(y, pred_proba, latency)
                
                model.learn_one(x, y)
                if i % settings.LOG_INTERVAL == 0:
                    stats = store.get_snapshot()
                    log_buffer.append({
                        "model": model.name,
                        "auc": stats["ROC-AUC"],
                        "acc": stats["Accuracy"],
                        "loss": stats["LogLoss"],
                        "lat": stats["Latency_us"]
                    })

            if i % settings.LOG_INTERVAL == 0:
                self._flush_logs(i, log_buffer)
                log_buffer = []
            
            if i % settings.CONSOLE_PRINT_INTERVAL == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                
                leader = max(
                    self.models, 
                    key=lambda m: self.metric_stores[m.name].auc.get()
                )
                leader_auc = self.metric_stores[leader.name].auc.get()
                
                logger.info(
                    f"Event {i} | Rate: {rate:.1f} ev/s | "
                    f"Leader: {leader.name} (AUC: {leader_auc:.4f})"
                )

        logger.info("Stream ended. Evaluation complete.")