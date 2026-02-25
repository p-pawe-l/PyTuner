from __future__ import annotations

import typing
import abc

from base.interfaces._env_factory import EnvFactory
from base.interfaces._evaluation_function import EvaluationFunction, EvaluationResult


@typing.runtime_checkable
class TrainableModel(typing.Protocol):

    @abc.abstractmethod
    def build(self, hyperparams: dict[str, typing.Any], envFactory: EnvFactory) -> 'TrainableModel':
        """
        Building the model from provided hyperparameters and environment factory.

        :param hyperparams: Hyperparameters for the model.
        :param envFactory: Factory to create the environment.
        :return: Self, for method chaining.
        """
        raise NotImplementedError("TrainableModel.build must be supported by the subclass")

    @abc.abstractmethod
    def train(self, timesteps: int, *args: typing.Any, **kwargs: typing.Any) -> 'TrainableModel':
        """
        Training the model.

        :param timesteps: Number of timesteps to train.
        :return: Self, for method chaining.
        """
        raise NotImplementedError("TrainableModel.train must be supported by the subclass")

    @abc.abstractmethod
    def evaluate(self, eval_func: EvaluationFunction, *args: typing.Any, **kwargs: typing.Any) -> EvaluationResult:
        """
        Evaluating the model.

        :param eval_func: Evaluation function to use.
        :return: The evaluation result.
        """
        raise NotImplementedError("TrainableModel.evaluate must be supported by the subclass")
