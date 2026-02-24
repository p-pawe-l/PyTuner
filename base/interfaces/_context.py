from __future__ import annotations
import typing
import dataclasses

@dataclasses.dataclass
class Context(typing.Protocol):
    """
    Context for the model.
    Mainly used for avoid rewriting to_dict method in each context class.
    """
    def to_dict(self) -> dict[str, typing.Any]:
        return dataclasses.asdict(self)

ContextType = typing.TypeVar("ContextType", bound=Context)
