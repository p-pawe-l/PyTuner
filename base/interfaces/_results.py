from __future__ import annotations
import typing

@typing.runtime_checkable
class TuningResult(typing.Protocol):
    def __repr__(self) -> str:
        ...
    
    @property
    def best_params(self) -> dict[str, typing.Any]:
        ...
    
    def save_to_file(self, file_path: str) -> None:
        ...
    
    def load_from_file(self, file_path: str) -> None:
        ...
    
    
    