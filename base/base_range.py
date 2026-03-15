import typing


from base.types import SamplerType, RangeType


class BaseRange(typing.Protocol):

    def __call__(self, sampler: SamplerType, name: str) -> RangeType:
        """
        Produce a sample of hyperparameters using the provided sampler.
        :param sampler: The sampler to use for producing the sample.
        :param name: The name of the hyperparameter to sample.
        :return: A sample of the hyperparameter.
        """
        raise NotImplementedError("BaseRange.__call__ must be supported by the subclass")
