from __future__ import annotations
import typing
import dataclasses

import optuna


@dataclasses.dataclass(frozen=True)
class Range_Interface(typing.Protocol):
    def __call__(self, trail: optuna.Trial, name: str) -> typing.Any:
        raise NotImplementedError("Range.__call__ must be supported by the subclass")
    

@dataclasses.dataclass(frozen=True)
class NumericalRange_Interface(Range_Interface):
    low: int | float 
    high: int | float  
    step: int | float | None = None 
    log: bool = False 

    def __call__(self, trail: optuna.Trial, name: str) -> int | float:
        return trail.suggest_int(name=name, 
                                 low=self.low, 
                                 high=self.high, 
                                 step=self.step, 
                                 log=self.log)
        
        
@dataclasses.dataclass(frozen=True)
class CategoricalRange_Interface(Range_Interface):
    choices: list[typing.Any]
    
    def __call__(self, trail: optuna.Trial, name: str) -> typing.Any:
        return trail.suggest_categorical(name, self.choices)
    
RangeType = NumericalRange_Interface | CategoricalRange_Interface