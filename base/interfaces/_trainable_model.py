from __future__ import annotations

import typing
import abc 

import gymnasium as gym 

@typing.runtime_checkable
class TrainableModel(typing.Protocol):
    
    @abc.abstractmethod
    def build(self, config: typing.Any, env: gym.Env) -> 'TrainableModel':
        """
        Building the model from provided configuration and environment.
        
        :param config: Configuration for the model.
        :param env: Environment to train the model on.
        :return: Self, for method chaining.
        """
        raise NotImplementedError("TrainableModel.build must be supported by the subclass")
    
    @abc.abstractmethod
    def train(self) -> 'TrainableModel':
        """
        Training the model.
        
        :return: Self, for method chaining.
        """
        raise NotImplementedError("TrainableModel.train must be supported by the subclass")
    
    @abc.abstractmethod
    def evaluate(self) -> 'TrainableModel':
        """
        Evaluating the model.
        
        :return: Self, for method chaining.
        """
        raise NotImplementedError("TrainableModel.evaluate must be supported by the subclass")
    
    


