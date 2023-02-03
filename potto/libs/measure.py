from dataclasses import dataclass
from numbers import Number
from numpy.random import uniform, triangular

from potto.lang.grammar import GExpr, Measure, TegVar, Const
from potto.lang.evaluate import evaluate


@dataclass(frozen=True)
class BoundedLebesgue(Measure):
    def __init__(self, lower_impl: Number | GExpr, upper_impl: Number | GExpr, tvar: TegVar):
        super().__init__(GExpr.to_gexpr(lower_impl), GExpr.to_gexpr(upper_impl), tvar)

    def sample(self, env, num_samples, gen_samples):
        return uniform(*self.get_bounds(env, num_samples, gen_samples))

    def get_bounds(self, env, num_samples, gen_samples):
        lower = evaluate(self.lower, env, num_samples, gen_samples)
        upper = evaluate(self.upper, env, num_samples, gen_samples)
        return lower, upper

    def density(self, _: float, env, num_samples, gen_samples) -> float:
        return 1

    def size(self):
        return self.upper - self.lower


@dataclass(frozen=True)
class Uniform(BoundedLebesgue):
    def __init__(self, lower_impl: Number | GExpr, upper_impl: Number | GExpr, tvar: TegVar):
        super().__init__(lower_impl, upper_impl, tvar)

    def density(self, _: float, env, num_samples, gen_samples) -> float:
        l, r = self.get_bounds(env, num_samples, gen_samples)
        return 1 / (r - l)

    def size(self):
        return Const(1)


@dataclass(frozen=True)
class Triangular(BoundedLebesgue):
    """A triangular distribution with mode 0 for simplicity."""

    def __init__(self, lower_impl: Number | GExpr, upper_impl: Number | GExpr, tvar: TegVar):
        super().__init__(lower_impl, upper_impl, tvar)

    def sample(self, env, num_samples, gen_samples):
        lower, upper = self.get_bounds(env, num_samples, gen_samples)
        assert lower < 0 and upper > 0
        return triangular(lower, 0, upper)
        # return uniform(lower, upper)

    def density(self, x: float, env, num_samples, gen_samples) -> float:
        l, r = self.get_bounds(env, num_samples, gen_samples)

        m = 0
        if l <= x <= m:
            return 2 * (x - l) / ((r - l) * (m - l))
        elif m <= x <= r:
            return 2 * (r - x) / ((r - l) * (r - m))
        else:
            return 0

    def size(self):
        return Const(1)
