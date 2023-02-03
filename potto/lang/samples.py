from dataclasses import dataclass, field
from collections.abc import Mapping
import random

from potto.lang.grammar import Sym
from potto.lang.traces import Trace


@dataclass(frozen=True)
class Sample:
    sample: float
    weight: float
    # trace: Trace | None


@dataclass(frozen=True)
class VarVal(Mapping):
    _data: Mapping[Sym, float] = field(default_factory=dict)  # This should not be modified after creation
    # TODO: copy input dict into instance variable?

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    # TODO: override dictionary joining to return VarVals instead of dicts


@dataclass(frozen=True)
class Samples:
    d: dict[Sym, Sample] = field(default_factory=dict)  # This should not be modified after creation
    # TODO: copy input dict into instance variable?

    def get_traceless_samples(self):
        d = {k: v if v.trace is None else Sample(v.sample, v.weight, None) for k, v in self.d.items()}
        return Samples(d)

    def __setitem__(self, name: Sym, value: Sample):
        raise AttributeError("Samples is immutable")

    def __getitem__(self, name: Sym) -> Sample:
        assert isinstance(name, Sym)
        return self.d[name]

    def __or__(self, other: "Samples") -> "Samples":
        u, v = set(self.d.keys()), set(other.d.keys())
        extended = self.d | other.d
        # Randomly select which trace to take
        for k in u & v:
            keep_self = random.choice([True, False])
            extended[k] = self.d[k] if keep_self else other.d[k]
        return Samples(extended)

    def __and__(self, other: "Samples") -> "Samples":
        u, v = set(self.d.keys()), set(other.d.keys())
        extended = self.d | other.d
        # For conflicts, select the traced variant
        for k in u & v:
            extended[k] = other.d[k] if self.d[k].trace is None else self.d[k]

            # left expression in a product and right expression in a product both are traced (they come from a delta)
            assert self.d[k].trace == None or other[k].trace == None, "Products of deltas are not supported"
        return Samples(extended)
