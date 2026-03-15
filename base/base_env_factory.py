from __future__ import annotations

import typing
import abc
import gymnasium


class BaseEnvFactory(abc.ABC):
    """
    Env factory for the model.
    """

    def __init__(self, task: str) -> None:
        self._task: str = task

    @abc.abstractmethod
    def create_env(self, *args: tuple, **kwargs: dict[str, typing.Any]) -> gymnasium.Env:
        """
        Creating the environment.

        :param task: The task to create the environment for.
        :return: The environment.
        """
        pass
