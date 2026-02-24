from __future__ import annotations

import typing 
import gymnasium

from base.interfaces._env_factory import EnvFactory

class GymnasiumEnvFactory(EnvFactory):
    """
    Gymnasium environment factory.
    
    This factory is used to create Gymnasium environments and vectorized environments.
    """
    def __init__(self, task: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        self._task: str = task
        self._args: typing.Any = args
        self._kwargs: typing.Any = kwargs
        
    def create_env(self) -> gymnasium.Env:
        """
        Creating a Gymnasium environment.
        
        :return: The created environment.
        """
        created_env: gymnasium.Env = gymnasium.make(self._task, *self._args, **self._kwargs)
        return created_env

    def create_vec_env(self) -> gymnasium.Env:
        """
        Creating a vectorized Gymnasium environment.
        
        :return: The created vectorized environment.
        """
        created_vec_env: gymnasium.Env = gymnasium.vector.make(self._task, *self._args, **self._kwargs)
        return created_vec_env