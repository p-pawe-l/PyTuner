from __future__ import annotations

from stable_baselines3.ppo import PPO

from stable_baselines3_tools._general_interface_model import _general_interface_model


class PPO_Model(_general_interface_model[PPO]):
    _algo_class = PPO
