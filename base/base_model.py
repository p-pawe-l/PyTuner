from __future__ import annotations

import typing
import abc

from base.types import EnvType, EvalFuncType, EvalResultType


class BaseModel(typing.Protocol):

    @abc.abstractmethod
    def build(self, hyperparams: dict[str, typing.Any], envFactory: EnvType) -> typing.Self:
        """
        Building the model from provided hyperparameters and environment factory.

        :param hyperparams: Hyperparameters for the model.
        :param envFactory: Factory to create the environment.
        :return: Self, for method chaining.
        """
        raise NotImplementedError("TrainableModel.build must be supported by the subclass")

    @abc.abstractmethod
    def train(self, timesteps: int, *args: typing.Any, **kwargs: typing.Any) -> typing.Self:
        """
        Training the model.

        :param timesteps: Number of timesteps to train.
        :return: Self, for method chaining.
        """
        raise NotImplementedError("TrainableModel.train must be supported by the subclass")

    @abc.abstractmethod
    def evaluate(self, eval_func: EvalFuncType, *args: typing.Any, **kwargs: typing.Any) -> EvalResultType:
        """
        Evaluating the model.

        :param eval_func: Evaluation function to use.
        :return: The evaluation result.
        """
        raise NotImplementedError("TrainableModel.evaluate must be supported by the subclass")
