from stable_baselines3.common.callbacks import BaseCallback


class BaseTuningCallback(BaseCallback):
    """
    Base callback for tuning.

    This callback is used to log the results of the tuning process.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # This method is called at every step of the training process.
        # You can add your logging logic here.
        return True
