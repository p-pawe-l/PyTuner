from __future__ import annotations

from base.interfaces._config import HyperparamsConfig
from stable_baselines3.a2c import A2C
from stable_baselines3_tools._general_interface_model import _general_interface_model

class A2C_Model(_general_interface_model[A2C]):
    _algo_class = A2C
    

        
    
