from __future__ import annotations
import typing
import dataclasses

from optuna_tools.ranges._interfaces import RangeType

@typing.runtime_checkable
@dataclasses.dataclass
class HyperparamsConfig(typing.Protocol):
    """
    There in subclasses we declare the hyperparams we want to tune.
    """
    
    def get_sample(self, trail: typing.Any) -> typing.Any:
        samples: dict[str, typing.Any] = {}
        
        for field in dataclasses.fields(self):
            range_obj: RangeType = getattr(self, field.name)
            samples[field.name] = range_obj(trail, field.name)
            
        return samples 