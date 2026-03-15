import abc

from stable_baselines3.common.callbacks import BaseCallback


class BaseTuningCallback(BaseCallback):
    """
    Base callback for tuning.

    This callback is used to log the results of the tuning process.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    @abc.abstractmethod
    def _on_tuning_single_run(self) -> bool:
        """Called at the end of each tuning run."""
        pass

    @abc.abstractmethod
    def on_tuning_start(self) -> bool:
        """Called at the beginning of the tuning process."""
        pass

    @abc.abstractmethod
    def on_tuning_end(self) -> bool:
        """Called at the end of the tuning process."""
        pass
