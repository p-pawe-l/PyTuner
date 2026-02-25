from __future__ import annotations

from stable_baselines3.dqn import DQN
from stable_baselines3_tools._general_interface_model import GeneralInterfaceModel

class DQN_Model(GeneralInterfaceModel[DQN]):
    _algo_class = DQN
    