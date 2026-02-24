from __future__ import annotations
import typing

import gymnasium 

from base.interfaces._trainable_model import TrainableModel
from base.interfaces._env_factory import EnvFactory

EvaluationResult = float | tuple[float, ...]

@typing.runtime_checkable
class EvaluationFunction(typing.Protocol):
    """
    Evaluation function for the model.
    
    Just a wrapper around the evaluation function.
    """
    def __call__(self, model: TrainableModel, env: gymnasium.Env, *args: typing.Any, **kwargs: typing.Any) -> EvaluationResult:
        """
        Evaluating the model.
        
        :param model: The model to evaluate.
        :param envFactory: The environment factory to create the environment for the evaluation.
        :param args: Additional arguments to the evaluation function.
        :param kwargs: Additional keyword arguments to the evaluation function.
        :return: The evaluation score.
        """
        raise NotImplementedError("EvaluationFunction.__call__ must be supported by the subclass")