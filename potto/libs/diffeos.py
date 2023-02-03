import itertools

from potto.lang.grammar import Var, TegVar, GExpr, Diffeomorphism
from potto.lang.evaluate import evaluate
from potto.libs.pmath import Sqrt


class Affine(Diffeomorphism):
    """
    f(x, y) = a * x + b * y + c, b * x - a * y + c
    f-1(x', y') = (a*x' + b*y' - c(a+b))/(a^2 + b^2), (b*x' - a*y' - c(b-a))/(a^2 + b^2)

    weight = | df(x, y)[0] / dx, df(x, y)[1] / dx |^-1
             | df(x, y)[0] / dy, df(x, y)[1] / dy |

    weight^(-1) = | a,  b |
                  | b, -a |

    |weight| = 1.0 / (a^2 + b^2)

    """

    def __str__(self):
        return f"{self.__class__.__name__}({self.vars, self.tvars})"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        a, b, c = vars
        x, y = tvars
        return (a * x + b * y + c, b * x - a * y + c)

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        a, b, c = vars
        x_, y_ = tvars
        return ((a * x_ + b * y_ - c * (a + b)) / (a * a + b * b), (b * x_ - a * y_ - c * (b - a)) / (a * a + b * b))

    def bounds_transfer(self, lower_left_corner: tuple, upper_right_corner: tuple, env):
        corners = itertools.product(*zip(lower_left_corner, upper_right_corner))
        corner_images = []
        for corner in corners:
            funs = self.function(self.vars, corner)
            corner_images.append([evaluate(f, env) for f in funs])
        mins = tuple(min(xs) for xs in zip(*corner_images))
        maxs = tuple(max(xs) for xs in zip(*corner_images))
        return mins, maxs


class SumDiffeo(Affine):
    """
    f(x, y) = x + y + c, x - y + c
    f-1(w, z) = (w + z - 2c)/2, (w - z) / 2
    """

    def __str__(self):
        return f"{self.__class__.__name__}({self.vars, self.tvars})"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        c, = vars
        x, y = tvars
        return x + y + c, x - y + c

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        c, = vars
        w, z = tvars
        return (w + z - 2 * c) / 2, (w - z) / 2


class QuadraticSquaredMinus1(Diffeomorphism):
    """
    f_t(x) = t^2 x^2 - 1 = (tx - 1)(tx + 1)
    f_t^{-1}(y) = sqrt(y + 1) / t
    """

    def __str__(self):
        return f"QuadraticSquaredMinus1({self.vars, self.tvars})"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        t, = vars
        x, = tvars
        return t * t * x * x - 1,

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        t, = vars
        y, = tvars
        return Sqrt(y + 1) / t,

    def bounds_transfer(self, lower_bounds: tuple, upper_bounds: tuple, env):
        lb, ub = lower_bounds[0], upper_bounds[0]
        f_lb = evaluate(self.function(self.vars, (lb,))[0], env)
        f_ub = evaluate(self.function(self.vars, (ub,))[0], env)
        lb, ub = evaluate(lb, env), evaluate(ub, env)
        if 0 <= lb:
            return (f_lb,), (f_ub,)
        elif ub <= 0:
            return (f_ub,), (f_lb,)
        else:
            return (-1,), (max(f_lb, f_ub),)


class SumOfSquares(Diffeomorphism):
    """
    f_t(x) = t^2 x^2 - 2tx + 1 = (tx - 1)^2
    f_t^{-1}(y) = (sqrt(y) + 1) / t
    """

    def __str__(self):
        return f"SumOfSquares({self.vars, self.tvars})"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        t, = vars
        x, = tvars
        return t * t * x * x - 2 * t * x + 1,

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        t, = vars
        y, = tvars
        return (Sqrt(y) + 1) / t,

    def bounds_transfer(self, lower_bound: tuple, upper_bound: tuple, env):
        raise NotImplementedError


class Scale2ShiftT(Diffeomorphism):
    """
    f_t(x) = 2 * x - t
    f_t^{-1}(y) = (y + t) / 2
    """

    def __str__(self):
        return f"{self.__class__.__name__}({self.vars, self.tvars})"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        t, = vars
        x, = tvars
        return 2 * x - t,

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        t, = vars
        y, = tvars
        return (y + t) / 2,

    def bounds_transfer(self, lower_bounds: tuple, upper_bounds: tuple, env):
        t = evaluate(self.vars[0], env)
        return (2 * lower_bounds[0] - t,), (2 * upper_bounds[0] - t,)


class ScaleByT(Diffeomorphism):
    """
    f_t(x) = t * x - 1
    f_t^{-1}(y) = (y - 1) / t
    """

    def __str__(self):
        return f"ScaleByT({self.vars, self.tvars})"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        t, = vars
        x, = tvars
        return t * x - 1,

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        t, = vars
        y, = tvars
        return (y + 1) / t,

    def bounds_transfer(self, lower_bounds: tuple, upper_bounds: tuple, env):
        t = evaluate(self.vars[0], env)
        return (t * lower_bounds[0] - 1,), (t * upper_bounds[0] - 1,)


class Shift(Affine):
    """
    f(x) = x + k = y
    f-1(y) = y - k = x
    """

    def __str__(self):
        # return f'[x + {self.vars[0].name}={self.vars[0].value}]'
        return f"({self.tvars[0].name} + {self.vars[0].name},)"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        return (tvars[0] + vars[0],)

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        return (tvars[0] - vars[0],)


class ShiftRight(Affine):
    """
    f(x) = x - k = y
    f-1(y) = y + k = x
    """

    def __str__(self):
        # return f'[x + {self.vars[0].name}={self.vars[0].value}]'
        return f"({self.tvars[0].name} - {self.vars[0].name},)"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        return (tvars[0] - vars[0],)

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        return (tvars[0] + vars[0],)


class FlipShift(Affine):
    """
    f(x) = k - x = y
    f-1(y) = k - y = x
    """

    def __str__(self):
        return "(k - x,)"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        return (vars[0] - tvars[0],)

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        return (vars[0] - tvars[0],)


class ShiftByT2D(Affine):
    """
    f1(x, y) = 4y - 2x + 3t
    f2(x, y) = y

    w = x + t
    z = y

    x = w - t
    y = z
    """

    def __str__(self):
        return "[x + t, y]"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        t, = vars
        x, y = tvars
        return x + t, y

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        t, = vars
        w, z = tvars
        return w - t, z


class SquareSlice2(Affine):
    """
    f1(x, y) = 4y - 2x + 3t
    f2(x, y) = y

    w = 4y - 2x + 3t
    z = y

    x = (4z - w + 3t) / 2
    y = z
    """

    def __str__(self):
        return "[y - 2x + t, y]"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        t, = vars
        x, y = tvars
        return 4 * y - 2 * x + 3 * t, y

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        t, = vars
        w, z = tvars
        return (4 * z - w + 3 * t) / 2, z


class SquareSlice(Affine):
    """
    f1(x, y) = y - 2x + t
    f2(x, y) = y
    [-2  1]   [-1/2 1/2] [0] = [y/2]
    [0   1]   [0      1] [y] = [y]

    w = y - 2x + t
    z = y
    2x = z - w + t
    """

    def __str__(self):
        return "[y - 2x + t, y]"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        t, = vars
        x, y = tvars
        return y - 2 * x + t, y

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        t, = vars
        w, z = tvars
        return (z - w + t) / 2, z


class OffAxisSquareSlice(Affine):
    """
    f1(x, y) = y - 2x
    f2(x, y) = 2y + x
    [-2  1]   [-2/5 1/5] [0] = [y/2]
    [1   2]   [1/5  2/5] [y] = [y]
    det A^{-1} = |-4/25 - 1/25| = 1/5
    """

    def __str__(self):
        return "[y - 2x, 2y + x]"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        return (vars[0] - 2 * tvars[0] + tvars[1], tvars[0] + 2 * tvars[1])

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        return (
            vars[0] + (-2 / 5) * tvars[0] + (1 / 5) * tvars[1],
            vars[0] + (1 / 5) * tvars[0] + (2 / 5) * tvars[1],
        )


class QuarterCircle(Diffeomorphism):
    """
    z1 = t - x^2 - y^2
    z2 = y

    x = sqrt(t - z1 - z2^2)
    y = z2
    """

    def __str__(self):
        return "[t - x^2 - y^2, y]"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        return (vars[0] - tvars[0] ** 2 - tvars[1] ** 2, tvars[1])

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        return (Sqrt(vars[0] - tvars[0] - tvars[1] ** 2), tvars[1])

    def bounds_transfer(self, lower_left_corner: tuple, upper_right_corner: tuple, env):
        corners = itertools.product(*zip(lower_left_corner, upper_right_corner))
        corner_images = []
        for corner in corners:
            funs = self.function(self.vars, corner)
            corner_images.append([evaluate(f, env) for f in funs])
        mins = tuple(min(xs) for xs in zip(*corner_images))
        maxs = tuple(max(xs) for xs in zip(*corner_images))
        return mins, maxs


class Simplex3D(Affine):
    """
    z1 = 1 - x - y - z
    z2 = y
    z3 = z

    x = 1 - y - z
    y = z2
    z = z3
    """

    def __str__(self):
        return "[1 - x - y - z, y, z]"

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        return (1 - tvars[0] - tvars[1] - tvars[2], tvars[1], tvars[2])

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        return (1 - tvars[1] - tvars[2], tvars[1], tvars[2])


class AboveDiag(Affine):
    def __str__(self):
        pass

    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        return (tvars[0] - vars[0] + tvars[1] - vars[1] + 0.75 + vars[2], tvars[1])

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        return (vars[0] - tvars[1] + vars[1] - 0.75 - vars[2], tvars[1])
