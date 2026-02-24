from __future__ import annotations

import typing

from stable_baselines3.common.base_class import BaseAlgorithm
import gymnasium as gym

from base.interfaces._trainable_model import TrainableModel
from base.interfaces._env_factory import EnvFactory
from base.interfaces._evaluation_function import EvaluationFunction
from base.interfaces._evaluation_function import EvaluationResult

StableBaselines3Model = typing.TypeVar("StableBaselines3Model", bound=BaseAlgorithm)


class _general_interface_model(TrainableModel, typing.Generic[StableBaselines3Model]):
    """
    General interface for Stable Baselines3 models.
    All Stable Baselines3 models are built on BaseAlgorithm interface.
    We can use that fact and build a general interface for all Stable Baselines3 models.

    Usage:
        class PPO_Model(_general_interface_model[PPO]):
            _algo_class = PPO
    """
    _algo_class: type[StableBaselines3Model]

    def __init__(self, policy: str = "MlpPolicy") -> None:
        self._policy: str = policy
        self._model: StableBaselines3Model | None = None
        self._env: gym.Env | None = None

    def build(self, config: dict[str, typing.Any], envFactory: EnvFactory) -> typing.Self:
        self._env = envFactory.create_env()
        self._model = self._algo_class(policy=self._policy, env=self._env, **config)
        return self

    def train(self, timesteps: int, *args: typing.Any, **kwargs: typing.Any) -> typing.Self:
        self._model.learn(total_timesteps=timesteps, *args, **kwargs)
        return self

    def evaluate(self, eval_func: EvaluationFunction, *args: typing.Any, **kwargs: typing.Any) -> EvaluationResult:
        result: EvaluationResult = eval_func(self._model, self._env, *args, **kwargs)
        return result
