from __future__ import annotations

from stable_baselines3.ppo import PPO

from stable_baselines3_tools._general_interface_model import GeneralInterfaceModel


class PPO_Model(GeneralInterfaceModel[PPO]):
    _algo_class = PPO
