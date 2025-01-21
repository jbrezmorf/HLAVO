from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class AbstractModel(ABC):
    def __init__(self, config, workdir=None):
        pass

    @abstractmethod
    def run(self, init_pressure: np.ndarray,  precipitation_value: float, model_params: dict, stop_time: float):
        pass

    @abstractmethod
    def get_data(self, time: float, data_name: str ="pressure") -> np.ndarray:
        pass

    @abstractmethod
    def get_times(self) -> list:
        pass

    @abstractmethod
    def get_space_step(self) -> float:
        pass

    @abstractmethod
    def plot_pressure(self):
        pass
