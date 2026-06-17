from abc import ABC, abstractmethod
from typing import Any
import pytorch_lightning as pl

class AudioTools(ABC):

    @abstractmethod
    def load_model(device)->pl.LightningModule:
        pass

    @abstractmethod
    def infer(waveform, sample_rate)->Any:
        pass
