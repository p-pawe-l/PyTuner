from __future__ import annotations

import typing

import numpy as np
import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm

from base.interfaces._evaluation_function import EvaluationFunction, EvaluationResult


class MeanRewardEvaluation(EvaluationFunction):
    """
    Evaluates a model by running N episodes and returning the mean total reward.
    """
    def __init__(self, n_episodes: int = 10) -> None:
        self._n_episodes: int = n_episodes

    def __call__(self, model: BaseAlgorithm, env: gym.Env, *args: typing.Any, **kwargs: typing.Any) -> EvaluationResult:
        episode_rewards: list[float] = []

        for _ in range(self._n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated

            episode_rewards.append(total_reward)

        return float(np.mean(episode_rewards))
