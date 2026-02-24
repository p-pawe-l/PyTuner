from __future__ import annotations

from stable_baselines3.sac import SAC

from stable_baselines3_tools._general_interface_model import _general_interface_model


class SAC_Model(_general_interface_model[SAC]):
    _algo_class = SAC
