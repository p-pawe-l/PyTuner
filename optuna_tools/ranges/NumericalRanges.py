from __future__ import annotations
import dataclasses

import optuna

from optuna_tools.ranges._interfaces import NumericalRange_Interface

@dataclasses.dataclass(frozen=True)
class IntRange(NumericalRange_Interface):
    low: int
    high: int
    step: int
    log: bool
    
    def __call__(self, trail: optuna.Trial, name: str) -> int:
        return trail.suggest_int(name=name, 
                                 low=self.low, 
                                 high=self.high, 
                                 step=self.step, 
                                 log=self.log)


@dataclasses.dataclass(frozen=True)
class FloatRange(NumericalRange_Interface):
    low: float
    high: float
    step: float
    log: bool
    
    def __call__(self, trail: optuna.Trial, name: str) -> float:
        return trail.suggest_float(name=name, 
                                   low=self.low, 
                                   high=self.high, 
                                   step=self.step, 
                                   log=self.log)