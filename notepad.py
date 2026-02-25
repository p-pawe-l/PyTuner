import dataclasses

from stable_baselines3_tools.models.TD3 import TD3_Model

from stable_baselines3_tools.callbacks.ComparisonCallback import ComparisonCallback

from gymnasium_tools.factory.GymnasiumEnvFactory import GymnasiumEnvFactory
from gymnasium_tools.eval_func.MeanRewardEvaluation import MeanRewardEvaluation

from optuna_tools.OptunaTuner import OptunaTuner
from optuna_tools.OptunaTunerContext import OptunaCreatingStudyContext, OptunaTuningContext, OptunaTunerContext
from optuna_tools.ranges.NumericalRanges import FloatRange, IntRange
from base.interfaces._config import HyperparamsConfig

@dataclasses.dataclass
class TD3Config(HyperparamsConfig):
    learning_rate: FloatRange = FloatRange(low=1e-5, high=1e-2, step=None, log=True)
    gamma: FloatRange = FloatRange(low=0.9, high=0.99, step=0.01, log=False)
    tau: FloatRange = FloatRange(low=0.001, high=0.01, step=0.001, log=False)


TASK = "Pendulum-v1"
TRAIN_TIMESTEPS = 1_000
TRAIN_ITERATIONS = 5


def main() -> None:
    model = TD3_Model(policy="MlpPolicy")
    env_factory = GymnasiumEnvFactory(task=TASK)
    eval_func = MeanRewardEvaluation()

    untuned_cb = ComparisonCallback(label="Untuned")
    untuned_hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 100,
    }
    model.build(untuned_hyperparams, env_factory)
    for _ in range(TRAIN_ITERATIONS):
        model.train(TRAIN_TIMESTEPS, callback=[untuned_cb])
    untuned_score = model.evaluate(eval_func)
    print(f"Untuned score: {untuned_score}")

    config = TD3Config()
    context = OptunaTunerContext(
        train_timesteps=10_000,
        creation_context=OptunaCreatingStudyContext(direction="maximize"),
        tuning_context=OptunaTuningContext(n_trials=50),
    )
    tuner = OptunaTuner(model, config, env_factory, eval_func, context)
    results = tuner.tune()
    print(f"Best params: {results.best_params}")

    tuned_cb = ComparisonCallback(label="Tuned")
    model.build(results.best_params, env_factory)
    for _ in range(TRAIN_ITERATIONS):
        model.train(TRAIN_TIMESTEPS, callback=[tuned_cb])
    tuned_score = model.evaluate(eval_func)
    print(f"Tuned score: {tuned_score}")

    ComparisonCallback.plot_comparison()


if __name__ == "__main__":
    main()
