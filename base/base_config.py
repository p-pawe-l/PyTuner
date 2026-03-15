from __future__ import annotations
import typing
import dataclasses

from base.types import SamplerType


class BaseConfig(typing.Protocol):
    """
    There in subclasses we declare the hyperparams we want to tune.
    """

    def produce_sample(self, sampler: SamplerType) -> dict[str, typing.Any]:
        """
        Produce a sample of hyperparameters using the provided sampler.
        :param sampler: The sampler to use for producing the sample.
        :return: A dictionary containing the sampled hyperparameters.
        """

        samples: dict[str, typing.Any] = {
            field.name: getattr(self, field.name)(sampler, field.name) for field in dataclasses.fields(self)
        }

        return samples

    def to_dict(self) -> dict[str, typing.Any]:
        """
        Convert the config to a dictionary.
        :return: A dictionary representation of the config.
        """
        return dataclasses.asdict(self)
