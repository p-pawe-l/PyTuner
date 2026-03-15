from __future__ import annotations

import abc
import typing

from types import ModelType, EnvType, EvalFuncType
from base_config import BaseConfig


class BaseTuningResults(abc.ABC):

    def __init__(self) -> None:
        self.best_params: typing.Optional[dict[str, typing.Any]] = None

    @abc.abstractmethod
    def save_to_file(self, file_path: str) -> None:
        """
        Saves tuning results to a file

        :param file_path: Path to the file where results will be saved
        """
        pass


class BaseTuner(abc.ABC):

    def __init__(self, model: ModelType, env: EnvType) -> None:
        self._model: ModelType = model
        self._env: EnvType = env

        self._is_preconfigured: bool = False

        self._eval_func: typing.Optional[EvalFuncType] = None
        self._config: typing.Optional[BaseConfig] = None
        self._args: typing.Optional[tuple] = None
        self._kwargs: typing.Optional[dict[str, typing.Any]] = None

    def pre_configure(
        self,
        eval_func: EvalFuncType,
        config: BaseConfig,
        *args: tuple,
        **kwargs: dict[str, typing.Any],
    ) -> None:
        """
        Pre configuration method to change:
        * evaluation function for the tuner
        * configuration for the tuner
        * additional positional arguments for the tuner
        * additional keyword arguments for the tuner
        """
        self._eval_func = eval_func
        self._config = config
        self._args = args
        self._kwargs = kwargs

        self._is_preconfigured = True

    @abc.abstractmethod
    def tune(
        self, runs: int = 20, *args: tuple, **kwargs: dict[str, typing.Any]
    ) -> None:
        pass
