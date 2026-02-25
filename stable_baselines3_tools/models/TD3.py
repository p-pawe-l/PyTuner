from __future__ import annotations

from stable_baselines3.td3 import TD3
from stable_baselines3_tools._general_interface_model import GeneralInterfaceModel

class TD3_Model(GeneralInterfaceModel[TD3]):
    _algo_class = TD3
    