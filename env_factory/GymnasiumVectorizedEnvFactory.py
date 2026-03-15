from __future__ import annotations

import typing
import gymnasium

from base.base_env_factory import BaseEnvFactory


class GymnasiumVectorizedEnvFactory(BaseEnvFactory):
    """
    Gymnasium vectorized environment factory.

    This factory is used to create vectorized Gymnasium environments.
    """

    def __init__(self, task: str) -> None:
        super().__init__(task)

    def create_env(
        self, *args: tuple, **kwargs: dict[str, typing.Any]
    ) -> gymnasium.Env:
        """
        Creating a vectorized Gymnasium environment.

        :return: The created vectorized environment.
        """
        created_env: gymnasium.Env = gymnasium.vector.make(self._task, *args, **kwargs)
        return created_env
