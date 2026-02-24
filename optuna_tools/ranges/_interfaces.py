from __future__ import annotations
import typing
import dataclasses

import optuna

@dataclasses.dataclass(frozen=True)
@typing.runtime_checkable
class NumericalRange_Interface(typing.Protocol):
    low: int | float 
    high: int | float  
    step: int | float | None = None 
    log: bool = False 
    
    def __call__(self, trail: optuna.Trial, name: str) -> int | float:
        raise NotImplementedError("NumericalRange.__call__ must be supported by the subclass")

    
        
@dataclasses.dataclass(frozen=True)
@typing.runtime_checkable
class CategoricalRange_Interface(typing.Protocol):
    choices: list[typing.Any]
    
    def __call__(self, trail: optuna.Trial, name: str) -> typing.Any:
        raise NotImplementedError("CategoricalRange.__call__ must be supported by the subclass")
    
RangeType = NumericalRange_Interface | CategoricalRange_Interface