from __future__ import annotations

import typing
import gymnasium

from base.interfaces._env_factory import EnvFactory

class GymnasiumVectorizedEnvFactory(EnvFactory): 
    def __init__(self, task: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        self._task: str = task
        self._args: typing.Any = args
        self._kwargs: typing.Any = kwargs

    def create_env(self) -> gymnasium.Env:
        created_env: gymnasium.Env = gymnasium.vector.make(self._task, *self._args, **self._kwargs)
        return created_env