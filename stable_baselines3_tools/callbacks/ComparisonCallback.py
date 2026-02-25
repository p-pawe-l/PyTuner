import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class ComparisonCallback(BaseCallback):
    """
    Callback that collects episode rewards during training.
    After both untuned and tuned runs, call `plot_comparison` to display
    a side-by-side chart of episode rewards.

    :param label: Label for this run (e.g. "Untuned" or "Tuned").
    :param update_interval: Refresh the plot every this many timesteps.
    """

    _runs: dict[str, list[float]] = {}

    def __init__(self, label: str, update_interval: int = 10) -> None:
        super().__init__()
        self._label = label
        self._update_interval = update_interval
        self._episode_rewards: list[float] = []

    def _on_training_start(self) -> None:
        self._episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
        return True

    def _on_training_end(self) -> None:
        ComparisonCallback._runs[self._label] = list(self._episode_rewards)

    @staticmethod
    def plot_comparison() -> None:
        """Display a chart comparing all recorded runs."""
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["tab:red", "tab:green", "tab:blue", "tab:orange"]

        for i, (label, rewards) in enumerate(ComparisonCallback._runs.items()):
            color = colors[i % len(colors)]
            episodes = list(range(1, len(rewards) + 1))
            ax.plot(episodes, rewards, alpha=0.3, color=color)

            window = max(1, len(rewards) // 20)
            smoothed = []
            for j in range(len(rewards)):
                start = max(0, j - window)
                smoothed.append(sum(rewards[start:j + 1]) / (j - start + 1))
            ax.plot(episodes, smoothed, color=color, linewidth=2, label=label)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.set_title("Untuned vs Tuned Hyperparameters")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
