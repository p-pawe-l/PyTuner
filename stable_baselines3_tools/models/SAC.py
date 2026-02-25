from __future__ import annotations

from stable_baselines3.sac import SAC

from stable_baselines3_tools._general_interface_model import GeneralInterfaceModel


class SAC_Model(GeneralInterfaceModel[SAC]):
    _algo_class = SAC
