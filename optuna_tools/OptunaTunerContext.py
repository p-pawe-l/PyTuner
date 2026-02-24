from __future__ import annotations

import typing
import dataclasses

import optuna 

from base.interfaces._context import Context

Direction = typing.Literal["maximize", "minimize"]

@dataclasses.dataclass
class OptunaCreatingStudyContext(Context):
    """Used for creating optuna`s study"""
    study_name: str = "OptunaTunerStudy"
    direction: Direction = "maximize"
    pruner: typing.Optional[optuna.pruners.BasePruner] = None 
    sampler: typing.Optional[optuna.samplers.BaseSampler] = None 
    storage: typing.Optional[optuna.storages.BaseStorage] = None 
    load_if_exists: bool = False
    directions: typing.Sequence[str | optuna.StudyDirection] | None = None
    

@dataclasses.dataclass
class OptunaTuningContext(Context):
    """Used for tuning optuna`s objective function during study"""
    n_trials: int | None = None
    timeout: float | None = None
    n_jobs: int = 1
    catch: typing.Iterable[type[Exception]] | type[Exception] = ()
    callbacks: typing.Iterable[typing.Callable[[optuna.Study, optuna.FrozenTrial], None]] | None = None
    gc_after_trial: bool = False
    show_progress_bar: bool = False
    
    def to_dict(self) -> dict[str, typing.Any]:
        return dataclasses.asdict(self)
    
    
@dataclasses.dataclass
class OptunaTunerContext:
    """Used for creating and tuning optuna`s study"""
    train_timesteps: int
    creation_context: OptunaCreatingStudyContext
    tuning_context: OptunaTuningContext
    