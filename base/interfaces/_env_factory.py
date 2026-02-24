from __future__ import annotations

import typing
import gymnasium as gym


@typing.runtime_checkable
class EnvFactory(typing.Protocol):
    """
    Env factory for the model.
    """
    def create_env(self) -> typing.Any:
        """
        Creating the environment.
        
        :param task: The task to create the environment for.
        :return: The environment.
        """
        raise NotImplementedError("EnvFactory.create_env must be supported by the subclass")
    
    def create_vec_env(self) -> typing.Any:
        """
        Creating the vectorized environment.
        
        :param task: The task to create the vectorized environment for.
        :return: The vectorized environment.
        """
        raise NotImplementedError("EnvFactory.create_vec_env must be supported by the subclass")