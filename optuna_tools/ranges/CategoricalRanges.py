from __future__ import annotations
import typing
import dataclasses

import optuna

from optuna_tools.ranges._interfaces import CategoricalRange_Interface

@dataclasses.dataclass(frozen=True)
class CategoricalRange(CategoricalRange_Interface):
    choices: list[typing.Any]
    
    def __call__(self, trail: optuna.Trial, name: str) -> typing.Any:
        return trail.suggest_categorical(name, self.choices)