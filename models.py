import abc
import logging
import math
from typing import Dict, Any, List, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as torch_optim

try:
    from river import linear_model, tree, facto, compose, preprocessing, optim, base
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}. Run 'pip install river'")

logger = logging.getLogger("Models")

class RobustHasher(base.Transformer):
    def __init__(self, n_features=1000):
        self.n_features = n_features

    def transform_one(self, x: Dict[str, Any]) -> Dict[str, float]:
        new_x = {}
        for key, value in x.items():
            feature_str = f"{key}={value}"
            hash_idx = abs(hash(feature_str)) % self.n_features
            new_x[f"f_{hash_idx}"] = 1.0
        return new_x

class OnlineModel(abc.ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.samples_seen = 0

    @abc.abstractmethod
    def learn_one(self, x: Dict[str, Any], y: int) -> None:
        pass

    @abc.abstractmethod
    def predict_one(self, x: Dict[str, Any]) -> float:
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name} | Seen: {self.samples_seen}>"

    def _build_robust_pipeline(self, model_step):
        return compose.Pipeline(
            compose.TransformerUnion(
                compose.SelectType(int, float) | preprocessing.StandardScaler(),
                compose.SelectType(str) | RobustHasher(n_features=1000)
            ),
            model_step
        )

class RiverLogisticRegression(OnlineModel):
    def __init__(self, name: str, lr: float = 0.01, l2: float = 0.0):
        super().__init__(name, {"lr": lr, "l2": l2})
        
        self.model = self._build_robust_pipeline(
            linear_model.LogisticRegression(
                optimizer=optim.SGD(lr=lr),
                l2=l2
            )
        )

    def learn_one(self, x: Dict[str, Any], y: int) -> None:
        self.model.learn_one(x, y)
        self.samples_seen += 1

    def predict_one(self, x: Dict[str, Any]) -> float:
        try:
            probs = self.model.predict_proba_one(x)
            return probs.get(1, 0.0)
        except Exception:
            return 0.5

class RiverHoeffdingTree(OnlineModel):
    def __init__(self, name: str, grace_period: int = 200, split_confidence: float = 1e-7):
        super().__init__(name, {"grace_period": grace_period, "delta": split_confidence})
        
        self.model = self._build_robust_pipeline(
            tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=grace_period,
                delta=split_confidence, 
                leaf_prediction='nb',
                nb_threshold=10
            )
        )

    def learn_one(self, x: Dict[str, Any], y: int) -> None:
        self.model.learn_one(x, y)
        self.samples_seen += 1

    def predict_one(self, x: Dict[str, Any]) -> float:
        try:
            probs = self.model.predict_proba_one(x)
            return probs.get(1, 0.0)
        except Exception:
            return 0.5

class RiverFM(OnlineModel):
    def __init__(self, name: str, n_factors: int = 10, weight_decay: float = 0.001):
        super().__init__(name, {"n_factors": n_factors, "l2_weight": weight_decay})
        
        self.model = self._build_robust_pipeline(
            facto.FFMClassifier(
                n_factors=n_factors,
                l2_weight=weight_decay,
                intercept=0.0,
                seed=42
            )
        )

    def learn_one(self, x: Dict[str, Any], y: int) -> None:
        self.model.learn_one(x, y)
        self.samples_seen += 1

    def predict_one(self, x: Dict[str, Any]) -> float:
        try:
            probs = self.model.predict_proba_one(x)
            return probs.get(1, 0.0)
        except Exception:
            return 0.5


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int]):
        super(SimpleMLP, self).__init__()
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2)) 
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class StreamingNeuralNet(OnlineModel):
    def __init__(self, name: str, input_dim: int = 128, hidden_layers: List[int] = [32, 16], lr: float = 0.01):
        super().__init__(name, {"input_dim": input_dim, "layers": hidden_layers, "lr": lr})
        
        self.device = torch.device("cpu")
        self.input_dim = input_dim
        
        self.hasher = RobustHasher(n_features=input_dim)
        
        self.model = SimpleMLP(input_dim, hidden_layers).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch_optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

    def _dict_to_tensor(self, x: Dict[str, Any]) -> torch.Tensor:
        hashed_dict = self.hasher.transform_one(x)
        vec = np.zeros(self.input_dim, dtype=np.float32)
        
        for key, val in hashed_dict.items():
            try:
                idx = int(key.split("_")[1])
                if 0 <= idx < self.input_dim:
                    vec[idx] = val
            except (IndexError, ValueError):
                continue
                
        return torch.from_numpy(vec).unsqueeze(0).to(self.device)

    def learn_one(self, x: Dict[str, Any], y: int) -> None:
        self.model.train()
        x_t = self._dict_to_tensor(x)
        y_t = torch.tensor([[float(y)]], dtype=torch.float32, device=self.device)
        
        self.optimizer.zero_grad()
        prediction = self.model(x_t)
        loss = self.criterion(prediction, y_t)
        loss.backward()
        self.optimizer.step()
        self.samples_seen += 1

    def predict_one(self, x: Dict[str, Any]) -> float:
        self.model.eval()
        with torch.no_grad():
            x_t = self._dict_to_tensor(x)
            prediction = self.model(x_t)
            return prediction.item()