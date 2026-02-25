import gymnasium
from stable_baselines3.common.callbacks import BaseCallback


class RenderEvaluationCallback(BaseCallback):
    """
    Callback that periodically renders the agent playing in the environment
    with the Gymnasium GUI (render_mode='human').

    Every ``eval_interval`` timesteps, a separate render environment is created
    and the agent plays ``n_episodes`` episodes visually.

    :param task: The Gymnasium task ID (e.g. "CartPole-v1").
    :param eval_interval: Run a visual evaluation every this many timesteps.
    :param n_episodes: Number of episodes to render each evaluation.
    """

    def __init__(self, task: str, eval_interval: int = 100, n_episodes: int = 1) -> None:
        super().__init__()
        self._task = task
        self._eval_interval = eval_interval
        self._n_episodes = n_episodes

    def _on_step(self) -> bool:
        if self.num_timesteps % self._eval_interval == 0:
            self._render_episodes()
        return True

    def _render_episodes(self) -> None:
        env = gymnasium.make(self._task, render_mode="human")
        for ep in range(self._n_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
            print(f"[Step {self.num_timesteps}] Render episode {ep + 1}/{self._n_episodes} — Reward: {total_reward:.2f}")
        env.close()
