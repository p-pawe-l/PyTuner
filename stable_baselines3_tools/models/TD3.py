from __future__ import annotations

from stable_baselines3.td3 import TD3
from stable_baselines3_tools._general_interface_model import _general_interface_model

class TD3_Model(_general_interface_model[TD3]):
    _algo_class = TD3
    