import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class TrainingProgressCallback(BaseCallback):
    """
    Callback that displays a live matplotlib window with training metrics.
    Shows episode rewards and episode lengths in real-time as training progresses.

    The window is created once and reused across multiple .train() calls,
    so the chart accumulates data without blocking.

    :param update_interval: Refresh the plot every ``update_interval`` timesteps.
    """

    def __init__(self, update_interval: int = 10) -> None:
        super().__init__()
        self._update_interval = update_interval
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._episode_numbers: list[int] = []
        self._episode_count = 0
        self._fig = None

    def _ensure_figure(self) -> None:
        if self._fig is not None and plt.fignum_exists(self._fig.number):
            return
        plt.ion()
        self._fig, (self._ax_reward, self._ax_length) = plt.subplots(2, 1, figsize=(8, 6))
        self._fig.suptitle("Training Progress")

        self._ax_reward.set_ylabel("Episode Reward")
        self._ax_reward.set_xlabel("Episode")
        self._line_reward, = self._ax_reward.plot([], [], color="tab:blue")

        self._ax_length.set_ylabel("Episode Length")
        self._ax_length.set_xlabel("Episode")
        self._line_length, = self._ax_length.plot([], [], color="tab:orange")

        self._fig.tight_layout()
        self._fig.show()

    def _on_training_start(self) -> None:
        self._ensure_figure()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_count += 1
                self._episode_numbers.append(self._episode_count)
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(int(info["episode"]["l"]))

        if self.num_timesteps % self._update_interval == 0 and self._episode_numbers:
            self._line_reward.set_data(self._episode_numbers, self._episode_rewards)
            self._ax_reward.relim()
            self._ax_reward.autoscale_view()

            self._line_length.set_data(self._episode_numbers, self._episode_lengths)
            self._ax_length.relim()
            self._ax_length.autoscale_view()

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

        return True
