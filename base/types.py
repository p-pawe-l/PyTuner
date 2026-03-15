from __future__ import annotations

import typing

import gymnasium
import stable_baselines3.common.base_class as sb3_base_class

from base.base_evaluation_function import BaseEvaluationFunction
from base.base_env_factory import BaseEnvFactory

SamplerType = typing.TypeVar("SamplerType")

ModelType = typing.TypeVar("ModelType")
EnvType = typing.TypeVar("EnvType", bound=gymnasium.Env | BaseEnvFactory)

EvalResultType = typing.TypeVar("EvalResultType", bound=float | tuple[float, ...])
EvalFuncType = typing.TypeVar(
    "EvalFuncType",
    bound=typing.Callable[
        [sb3_base_class.BaseAlgorithm, gymnasium.Env, tuple, dict[str, typing.Any]],
        EvalResultType,
    ]
    | typing.Callable[[sb3_base_class.BaseAlgorithm, gymnasium.Env], EvalResultType]
    | BaseEvaluationFunction,
)
