from __future__ import annotations

import typing
import gymnasium as gym

from base.interfaces._trainable_model import TrainableModel
from base.interfaces._env_factory import EnvFactory
from base.interfaces._results import TuningResult
from base.interfaces._config import HyperparamsConfig
from base.interfaces._evaluation_function import EvaluationFunction
from base.interfaces._context import ContextType

@typing.runtime_checkable
class Tuner(typing.Protocol):
    """
    Tuner for the model.
    """
    def __init__(self, model: TrainableModel, config: HyperparamsConfig, envFactory: EnvFactory, 
                 evaluationFunction: EvaluationFunction, context: ContextType) -> None:
        raise NotImplementedError("Tuner.__init__ must be supported by the subclass")
    
    def tune(self) -> TuningResult:
        raise NotImplementedError("Tuner.tune must be supported by the subclass")