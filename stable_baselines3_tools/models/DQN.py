from __future__ import annotations

from stable_baselines3.dqn import DQN
from stable_baselines3_tools._general_interface_model import _general_interface_model

class DQN_Model(_general_interface_model[DQN]):
    _algo_class = DQN
    