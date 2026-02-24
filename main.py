import dataclasses

from base.interfaces._trainable_model import TrainableModel
from base.interfaces._env_factory import EnvFactory
from base.interfaces._config import HyperparamsConfig

from optuna_tools.OptunaTunerContext import OptunaCreatingStudyContext, OptunaTuningContext, OptunaTunerContext
from optuna_tools.OptunaTuner import OptunaTuner

from stable_baselines3_tools.models.PPO import PPO_Model
from gymnasium_tools.GymnasiumEnvFactory import GymnasiumEnvFactory
from optuna_tools.ranges.NumericalRanges import FloatRange
from gymnasium_tools.MeanRewardEvaluation import MeanRewardEvaluation


def prepare_context() -> OptunaTunerContext:
    creation_context: OptunaCreatingStudyContext = OptunaCreatingStudyContext()
    tuning_context: OptunaTuningContext = OptunaTuningContext(
        n_trials=10,
    )
    return OptunaTunerContext(train_timesteps=1000, 
                              creation_context=creation_context, 
                              tuning_context=tuning_context)
    
def main() -> None:
    context: OptunaTunerContext = prepare_context() 
    model: TrainableModel = PPO_Model()
    env_factory: EnvFactory = GymnasiumEnvFactory(task="CartPole-v1")
    
    @dataclasses.dataclass
    class PPOConfig(HyperparamsConfig):
        learning_rate: FloatRange = FloatRange(0.001, 0.01, 0.001, False)

    tuner = OptunaTuner(model=model, 
                        config=PPOConfig(), 
                        envFactory=env_factory, 
                        evaluationFunction=MeanRewardEvaluation(), 
                        context=context)
    
    tuning_results = tuner.tune()
    print(tuning_results)
    
if __name__ == "__main__":
    main()