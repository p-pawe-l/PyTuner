from __future__ import annotations

import typing
import optuna 

from base.interfaces._results import TuningResult

class OptunaTuningResults(TuningResult):
    def __init__(self, study: optuna.Study) -> None:
        self._study: optuna.Study = study
        self._best_params: dict[str, typing.Any] = study.best_params
        
    def __repr__(self) -> str:
        return f"OptunaTuningResults(best_params={self._best_params})"
        
    @property
    def best_params(self) -> dict[str, typing.Any]:
        return self._best_params
    
    
        