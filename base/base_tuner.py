from __future__ import annotations

import abc
import typing
import gymnasium

import enum

from base.base_tuning_callback import BaseTuningCallback
from base.types import EvalFuncType
from base.base_model import BaseModel
from base.base_config import BaseConfig

from errors.tuner_errros import NotPreConfiguredTunerError


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

    class CallbackOption(enum.Enum):
        TUNE_START = "tune_start"
        TUNE_END = "tune_end"

    def __init__(self, model: BaseModel, env: gymnasium.Env) -> None:
        self._model: BaseModel = model
        self._env: gymnasium.Env = env

        self._is_preconfigured: bool = False

        self._eval_func: typing.Optional[EvalFuncType] = None
        self._config: typing.Optional[BaseConfig] = None
        self._args: typing.Optional[tuple] = None
        self._kwargs: typing.Optional[dict[str, typing.Any]] = None

        self._callbacks: typing.Optional[list[BaseTuningCallback]] = []

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

    def set_callbacks(self, callbacks: list[BaseTuningCallback] | BaseTuningCallback) -> None:
        if isinstance(callbacks, list):
            self._callbacks = callbacks
        else:
            self._callbacks = [callbacks]

    @abc.abstractmethod
    def _custom_tune_method(self, runs: int = 20, *args: tuple, **kwargs: dict[str, typing.Any]) -> BaseTuningResults:
        """
        Custom tuning method to be implemented by the subclass.

        :param runs: Number of tuning runs to perform.
        :param args: Additional positional arguments for the tuning method.
        :param kwargs: Additional keyword arguments for the tuning method.
        :return: The tuning results.
        """
        pass

    def __run_callbacks(self, option: CallbackOption) -> None:
        if self._callbacks is not None:
            for callback in self._callbacks:
                match option:
                    case self.CallbackOption.TUNE_START:
                        callback.on_tuning_start(self)
                    case self.CallbackOption.TUNE_END:
                        callback.on_tuning_end(self)

    def tune(self, runs: int = 20, *args: tuple, **kwargs: dict[str, typing.Any]) -> BaseTuningResults:
        if not self._is_preconfigured:
            raise NotPreConfiguredTunerError("Tuner must be pre-configured before tuning")

        self.__run_callbacks(self.CallbackOption.TUNE_START)

        results = self._custom_tune_method(runs, *args, **kwargs)

        self.__run_callbacks(self.CallbackOption.TUNE_END)

        return results
