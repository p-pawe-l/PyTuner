from __future__ import annotations
import typing
import dataclasses

import pytest
import optuna

from base.interfaces._config import HyperparamsConfig
from optuna_tools.ranges.NumericalRanges import FloatRange, IntRange
from optuna_tools.ranges.CategoricalRanges import CategoricalRange

@pytest.fixture
def config() -> HyperparamsConfig:
    # Testing all possible types of ranges
    # We need to mock example config object for tests 
    @dataclasses.dataclass
    class TestConfig(HyperparamsConfig):
        learning_rate: FloatRange = FloatRange(0.01, 0.02, 0.005, False)
        n_steps: IntRange = IntRange(100, 200, 10, False)
        batch_size: CategoricalRange = CategoricalRange(choices=[16, 32, 64, 128])

    return TestConfig()

@pytest.fixture
def trial() -> optuna.Trial:
    # We need to mock trail object for tests 
    study: optuna.Study = optuna.create_study()
    return study.ask()

def test_config_has_sample_method(config: HyperparamsConfig) -> None:
    assert hasattr(config, "get_sample")
    
def test_config_has_fields(config: HyperparamsConfig) -> None:
    assert hasattr(config, "learning_rate")
    assert hasattr(config, "n_steps")
    assert hasattr(config, "batch_size")
    
def test_float_range_sampling(config: HyperparamsConfig, trial: optuna.Trial) -> None:
    samples: dict[str, typing.Any] = config.get_sample(trial)
    assert isinstance(samples["learning_rate"], float)
    assert 0.01 <= samples["learning_rate"] <= 0.02
    
def test_int_range_sampling(config: HyperparamsConfig, trial: optuna.Trial) -> None:
    samples: dict[str, typing.Any] = config.get_sample(trial)
    assert isinstance(samples["n_steps"], int)
    assert 100 <= samples["n_steps"] <= 200
    
def test_categorical_range_sampling(config: HyperparamsConfig, trial: optuna.Trial) -> None:
    samples: dict[str, typing.Any] = config.get_sample(trial)
    assert isinstance(samples["batch_size"], int)
    assert samples["batch_size"] in [16, 32, 64, 128]

