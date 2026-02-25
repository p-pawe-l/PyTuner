from __future__ import annotations
import typing

class InvalidHyperparamError(Exception):
    def __init__(self, hyperparams: dict[str, typing.Any]) -> None:
        self.hyperparams = hyperparams
        super().__init__(f"Invalid hyperparams provided: {hyperparams}")
        
        
class InvalidHyperparamTypeError(InvalidHyperparamError):
    def __init__(self, hyperparams: dict[str, typing.Any]) -> None:
        self.hyperparams = hyperparams
        super().__init__(f"Invalid hyperparams type provided: {type(hyperparams)}")