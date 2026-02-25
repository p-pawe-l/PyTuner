from __future__ import annotations

from stable_baselines3.a2c import A2C
from stable_baselines3_tools._general_interface_model import GeneralInterfaceModel

class A2C_Model(GeneralInterfaceModel[A2C]):
    _algo_class = A2C
    

        
    
