from __future__ import annotations
import typing

import gymnasium
import stable_baselines3.common.base_class as sb3_base_class

if typing.TYPE_CHECKING:
    from base.types import EvalResultType


class BaseEvaluationFunction(typing.Protocol):
    """
    Evaluation function for the model.

    Just a wrapper around the evaluation function.
    """

    def __call__(
        self,
        model: sb3_base_class.BaseAlgorithm,
        env: gymnasium.Env,
        *args: tuple,
        **kwargs: dict[str, typing.Any],
    ) -> EvalResultType:
        """
        Evaluating the model.

        :param model: The model to evaluate.
        :param env: The environment to use for the evaluation.
        :param args: Additional arguments to the evaluation function.
        :param kwargs: Additional keyword arguments to the evaluation function.
        :return: The evaluation score.
        """
        raise NotImplementedError("EvaluationFunction.__call__ must be supported by the subclass")
