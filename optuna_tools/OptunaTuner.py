from __future__ import annotations

import typing
import optuna
from sympy.polys import resultant

from base.interfaces._tuner import Tuner
from base.interfaces._trainable_model import TrainableModel
from base.interfaces._env_factory import EnvFactory
from base.interfaces._evaluation_function import EvaluationFunction
from base.interfaces._evaluation_function import EvaluationResult
from base.interfaces._config import HyperparamsConfig
from base.interfaces._results import TuningResult

from optuna_tools.OptunaTunerContext import OptunaCreatingStudyContext, OptunaTuningContext, OptunaTunerContext
from optuna_tools.OptunaTuningResults import OptunaTuningResults

class OptunaTuner(Tuner):
    """Optuna Tuner for Stable Baselines3 models learning on gymnasium environments"""
    def __init__(self, model: TrainableModel, config: HyperparamsConfig, envFactory: EnvFactory, 
                 evaluationFunction: EvaluationFunction, context: OptunaTunerContext) -> None:
        self._model: TrainableModel = model
        self._config: HyperparamsConfig = config
        self._envFactory: EnvFactory = envFactory
        self._eval_func: EvaluationFunction = evaluationFunction
        
        self._creation_context: OptunaCreatingStudyContext = context.creation_context
        self._tuning_context: OptunaTuningContext = context.tuning_context
        
        self._train_timesteps: int = context.train_timesteps

    # Special method for optuna to optimize during study 
    def _objective(self, trail: optuna.Trial) -> float:
        # Getting sample of hyperparams from configured ranges of hyperparams
        sample_hyperparams: dict[str, typing.Any] = self._config.get_sample(trail)
        
        # Building -> Training -> Evaluating Model
        # Some kind of builder pattern 
        result: EvaluationResult = self._model \
            .build(sample_hyperparams, self._envFactory) \
            .train(timesteps=self._train_timesteps) \
            .evaluate(self._eval_func)
        
        return result
    
    def tune(self) -> TuningResult:
        study: optuna.Study = optuna.create_study(**self._creation_context.to_dict())
        study.optimize(self._objective, **self._tuning_context.to_dict())
        
        return OptunaTuningResults(study)
