from __future__ import annotations
import typing

import pytest 

from base.interfaces._env_factory import EnvFactory
from base.interfaces._evaluation_function import EvaluationFunction

from stable_baselines3_tools._general_interface_model import GeneralInterfaceModel

from stable_baselines3_tools.errors.InvalidHyperparamError import InvalidHyperparamTypeError
from stable_baselines3_tools.errors.InvalidNumberOfTimestepsError import InvalidNumberOfTimestepsError
from stable_baselines3_tools.errors.InvalidEvaluationFunctionError import InvalidEvaluationFunctionError

@pytest.fixture 
def model() -> GeneralInterfaceModel:
    return GeneralInterfaceModel()

@pytest.fixture
def valid_hyperparams() -> dict[str, typing.Any]:
    return {
        "learning_rate": 0.001,
        "n_steps": 100,
        "batch_size": 32,
    }
@pytest.fixture
def invalid_hyperparams() -> list[int]:
    return [0.001, 100, 32]

@pytest.fixture
def env_factory() -> EnvFactory:
    return EnvFactory()

@pytest.fixture
def eval_func() -> EvaluationFunction:
    return EvaluationFunction()

@pytest.fixture
def invalid_eval_func() -> EvaluationFunction:
    return "invalid_eval_func"


def test_build_method(model: GeneralInterfaceModel, valid_hyperparams: dict[str, typing.Any], 
                      env_factory: EnvFactory, invalid_hyperparams: list[int]) -> None:
    model.build(valid_hyperparams, env_factory)
    assert model._model is not None 
    
    with pytest.raises(InvalidHyperparamTypeError):
        model.build(invalid_hyperparams, env_factory)
        assert model._model is None 
        
def test_train_method(model: GeneralInterfaceModel, valid_hyperparams: dict[str, typing.Any], 
                      env_factory: EnvFactory, invalid_hyperparams: list[int]) -> None:
    model.build(valid_hyperparams, env_factory)
    model.train(100)
    assert model._model is not None 
    
    with pytest.raises(InvalidNumberOfTimestepsError):
        model.train(0)
        assert model._model is None 
        
def test_evaluate_method(model: GeneralInterfaceModel, valid_hyperparams: dict[str, typing.Any], 
                         env_factory: EnvFactory, eval_func: EvaluationFunction) -> None:
    model.build(valid_hyperparams, env_factory)
    model.evaluate(eval_func)
    assert model._model is not None 
    
    with pytest.raises(InvalidEvaluationFunctionError):
        model.evaluate(invalid_eval_func)
        assert model._model is None 