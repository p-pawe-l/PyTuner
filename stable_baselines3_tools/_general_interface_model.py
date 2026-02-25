from __future__ import annotations

import typing

import gymnasium
from stable_baselines3.common.base_class import BaseAlgorithm

from base.interfaces._trainable_model import TrainableModel
from base.interfaces._env_factory import EnvFactory
from base.interfaces._evaluation_function import EvaluationFunction
from base.interfaces._evaluation_function import EvaluationResult

from stable_baselines3_tools.errors.InvalidHyperparamError import InvalidHyperparamTypeError
from stable_baselines3_tools.errors.InvalidNumberOfTimestepsError import InvalidNumberOfTimestepsError
from stable_baselines3_tools.errors.MissingModelError import MissingModelError
from stable_baselines3_tools.errors.InvalidEvaluationFunctionError import InvalidEvaluationFunctionError
from stable_baselines3_tools.errors.NotCallableError import NotCallableError

StableBaselines3Model = typing.TypeVar("StableBaselines3Model", bound=BaseAlgorithm)


class GeneralInterfaceModel(TrainableModel, typing.Generic[StableBaselines3Model]):
    """
    General interface for Stable Baselines3 models.
    All Stable Baselines3 models are built on BaseAlgorithm interface.
    We can use that fact and build a general interface for all Stable Baselines3 models.

    Usage:
        class PPO_Model(GeneralInterfaceModel[PPO]):
            _algo_class = PPO
    """
    _algo_class: type[StableBaselines3Model]
    
    # Adding this functions enables easier testing
    def _validate_model(self) -> None:
        if self._model is None: raise MissingModelError("Build model before training or evaluating")
        else: return None
    
    def _validate_timesteps(self, timesteps: int) -> None:
        if timesteps <= 0 : raise InvalidNumberOfTimestepsError(timesteps)
        else: return None
    
    def _validate_hyperparams(self, hyperparams: dict[str, typing.Any]) -> None:
        if not isinstance(hyperparams, dict): raise InvalidHyperparamTypeError(hyperparams)
        else: return 
        
    def _validate_eval_func(self, eval_func: EvaluationFunction) -> None:
        if not isinstance(eval_func, EvaluationFunction): raise InvalidEvaluationFunctionError(eval_func)
        if not callable(eval_func): raise NotCallableError(eval_func)
        else: return None
    
            
    """-----------------------------API INTERFACE METHODS TO USE FOR BUILDING, TRAINING AND EVALUATING THE MODEL-----------------------------"""

    def build(self, hyperparams: dict[str, typing.Any], envFactory: EnvFactory, policy: str = "MlpPolicy") -> typing.Self:
        """
        Building the model from provided hyperparameters and environment factory.
        
        :param hyperparams: Hyperparameters for the model.
        :param envFactory: Environment factory to create the environment.
        :param policy: Policy to use for the model.
        """
        self._validate_hyperparams(hyperparams)
        
        env: gymnasium.Env = envFactory.create_env()
        self._model = self._algo_class(policy=policy, env=env, **hyperparams)
        return self

    def train(self, timesteps: int, *args: typing.Any, **kwargs: typing.Any) -> typing.Self:
        """
        Training the model for a given number of timesteps.
        
        :param timesteps: Number of timesteps to train.
        :param args: Additional arguments to the training function.
        :param kwargs: Additional keyword arguments to the training function.
        :return: Self, for method chaining.
        """
        self._validate_model()
        self._validate_timesteps(timesteps)

        self._model.learn(total_timesteps=timesteps, *args, **kwargs)
        return self

    def evaluate(self, eval_func: EvaluationFunction, *args: typing.Any, **kwargs: typing.Any) -> EvaluationResult:
        """
        Evaluating the model using a given evaluation function.
        
        :param eval_func: Evaluation function to use.
        :param args: Additional arguments to the evaluation function.
        :param kwargs: Additional keyword arguments to the evaluation function.
        :return: Evaluation result.
        """
        self._validate_model()
        self._validate_eval_func(eval_func)

        result: EvaluationResult = eval_func(self._model, self._model.get_env(), *args, **kwargs)
        return result
