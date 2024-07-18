from dataclasses import dataclass
import math

from potto.lang.grammar import GExpr, Const, UnaryBuiltin, Diffeomorphism, heaviside as Heaviside


@dataclass(frozen=True)
class Sqrt(UnaryBuiltin):
    def eval(self, evaluate_helper, env):
        try:
            return math.sqrt(evaluate_helper(self.expr, env))
        except ValueError:
            return math.nan

    def deriv(self, derivative, context):
        return 1 / (2 * Sqrt(derivative(self.expr, context)))

    def __str__(self):
        return f"sqrt({self.expr})"


@dataclass(frozen=True)
class Sqr(UnaryBuiltin):
    def eval(self, evaluate_helper, env):
        return math.pow(evaluate_helper(self.expr, env), 2)

    def deriv(self, derivative, context):
        return derivative(self.expr, context)

    def __str__(self):
        return f"sqr({self.expr})"


@dataclass(frozen=True)
class Exp(UnaryBuiltin):
    def eval(self, evaluate_helper, env):
        return math.exp(evaluate_helper(self.expr, env))

    def deriv(self, derivative, context):
        return Exp(self.expr) * derivative(self.expr, context)

    def __str__(self):
        return f"exp({self.expr})"


@dataclass(frozen=True)
class Abs(UnaryBuiltin):
    expr: GExpr

    def eval(self, evaluate_helper, env):
        return abs(evaluate_helper(self.expr, env))

    def deriv(self, derivative, context):
        # abs(x) = 1 if x > 0 else -1
        class NotDiffeo(Diffeomorphism):
            expr = self.expr

            def function(self, vars, tvars):
                return self.expr

            def inverse(self, vars, tvars):
                pass

        class AlsoNotDiffeo(Diffeomorphism):
            expr = self.expr

            def function(self, vars, tvars):
                return -self.expr

            def inverse(self, vars, tvars):
                pass

        # TODO: make a heaviside that doesn't need a diffeomorphism
        gt0 = Heaviside(NotDiffeo(tuple(), tuple(), Const(0)))
        leq0 = -Heaviside(AlsoNotDiffeo(tuple(), tuple(), Const(0)))
        return (gt0 + leq0) * derivative(self.expr, context)

    def __str__(self):
        return f"abs({self.expr})"
