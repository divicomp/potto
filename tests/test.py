import unittest
from unittest import TestCase
import numpy as np
import random

from potto import (
    Shift,
    FlipShift,
    SquareSlice,
    QuarterCircle,
    Simplex3D,
    OffAxisSquareSlice,
    ShiftRight,
    ScaleByT,
    Scale2ShiftT,
)
from potto import (
    Delta,
    Diffeomorphism,
    IfElse,
    Var,
    TegVar,
    heaviside as Heaviside,
    GExpr,
    Const,
    Int,
    Sym,
    App,
    Function,
)
from potto import Sqrt
from potto import deriv
from potto import evaluate_all
from potto import VarVal
from potto import BoundedLebesgue, Triangular, Uniform, Affine
from potto.lang.traces import TraceName, Trace
from potto.lang.evaluate_utils import TraceEnv, to_env, to_irenv, Gen
from potto.lang.evaluate import generate, evaluate
from potto.libs.diffeos import SumOfSquares, QuadraticSquaredMinus1, SquareSlice2, ShiftByT2D

random.seed(0)
np.random.seed(0)


class TestSimple(TestCase):
    def test_linear(self):
        x = Var("x")
        three_x = x + x + x
        v = evaluate(three_x, env_or_var_val=VarVal({x.name: 1}))
        self.assertEqual(v, 3)

    def test_multiply(self):
        x = Var("x")
        cube = x * x * x
        self.assertEqual(evaluate(cube, env_or_var_val=VarVal({x.name: 2})), 8)

    def test_polynomial(self):
        x = Var("x")
        poly = x * x * x + x * x + x
        self.assertAlmostEqual(evaluate(poly, env_or_var_val=VarVal({x.name: 2})), 14, places=3)

    def test_curry(self):
        # from potto.print import get_printer
        z = Var("z")
        dz = Var("dz")
        f = Var("f")
        df = Var("df")
        g = Var("g")
        # f(g(x))
        # \f.\g.\x.f(g(x))
        # a = Sqrt(Sqrt(f))
        app = lambda x, *xs: App(x, xs)
        fgx = app(f, app(g, app(f, z)))
        # fgx = App(f, (App(g, (App(f, (z,)),))
        a = Function((f,), Function((g,), fgx))
        # o = get_printer(a)
        # o.println(Sym('h'))
        # print(str(o))
        # Function((f,df),
        #          body=Function((g, dg),
        #             body=App(df,
        #                 args=(App(g, App(f, z),
        #                        App(function=Var(name=Sym(dg, 11)),
        #                            args=(App(function=Var(name=Sym(f, 2)), args=(Var(name=Sym(z, 0)),), name=Sym(, 5)),
        #                            App(function=Var(name=Sym(df, 10)), args=(Var(name=Sym(z, 0)), Var(name=Sym(dz, 1))),
        #                                name=Sym(, 5))), name=Sym(,
        #          6))))))

        # print(repr(a))
        deriv_out = deriv(a, {z.name: dz.name})

        # print(repr(deriv_out))
        # \f. \g. f(g(z))
        # \f, df. \g, dg. (df(g(z), dg(z,0)))
        # Function((f, df),
        #          body=Function((g, dg),
        #                        body=App(df,
        #                                 args=
        #                                 (App(g, z),
        #                                  App(dg, z, Const(value=0))))))
        # self.assertAlmostEqual(evaluate(poly, env_or_var_val=VarVal({x.name: 2})), 14, places=3)


class TestSampling(TestCase):
    def test_sample_x(self):
        x = TegVar("x")
        mu = BoundedLebesgue(0, 1, x)
        e = Int(x, mu)
        self.assertAlmostEqual(evaluate(e, num_samples=1000), 0.5, places=1)

    def test_sample_x_plus_1(self):
        x = TegVar("x")
        mu = BoundedLebesgue(0, 1, x)
        e = Int(x + 1, mu)
        self.assertAlmostEqual(evaluate(e, num_samples=1000), 1.5, places=1)

    def test_sample_x_times_y(self):
        x = TegVar("x")
        y = TegVar("y")
        mu = BoundedLebesgue(0, 1, x)
        nu = BoundedLebesgue(0, 1, y)
        e = Int(Int(x * y, mu), nu)
        self.assertAlmostEqual(evaluate(e, num_samples=1000), 0.25, places=1)


class TestEliminateEquivalences(TestCase):
    def setUp(self):
        self.tvx = TegVar("x")
        self.tvy = TegVar("y")
        self.k = Var("k")
        self.l = Var("l")
        self.m = Var("m")
        self.shiftl = Shift((self.l,), (self.tvx,))
        self.shiftk = Shift((self.k,), (self.tvx,))
        self.shiftm = Shift((self.m,), (self.tvx,))
        self.flip_shiftk = FlipShift((self.k,), (self.tvx,))
        self.mu_m1to1 = BoundedLebesgue(-1, 1, self.tvx)
        self.mu = BoundedLebesgue(0, 1, self.tvx)

    def test_delta_times_k_plus_1(self):
        delta = Delta(self.shiftk)
        expr = delta * (self.k + 1)
        e = Int(expr, self.mu_m1to1)
        # int_-1^1 delta(x) dmu(x)
        var_val = VarVal({self.k.name: 0})
        v = evaluate(e, var_val)
        self.assertEqual(v, 1)

        var_val = VarVal({self.k.name: 0.7})
        # int_-1^1 delta(x + 0.5) dmu(x)
        v = evaluate(e, var_val)
        self.assertEqual(v, 1.7)

    def test_delta_elimination(self):
        delta = Delta(self.shiftk)
        heaviside = Heaviside(self.shiftl)
        var_val = VarVal({self.k.name: 0, self.l.name: 1 / 2})

        # int_-1^1 delta(x) Heaviside(x + 1/2) dmu(x) = 0
        e = Int(delta * heaviside, self.mu_m1to1)
        v = evaluate(e, env_or_var_val=var_val)
        self.assertEqual(v, 1)

        # int_-1^1 delta(x) Heaviside(x - 1/2) dmu(x) = 1
        var_val = VarVal({self.k.name: 0, self.l.name: -1 / 2})
        v = evaluate(e, env_or_var_val=var_val)
        self.assertEqual(v, 0)

    def test_shifted_delta(self):
        # \int_x delta(x + 1) H(x) = 0
        delta = Delta(self.shiftk)
        heaviside = Heaviside(self.shiftl)
        e = Int(delta * heaviside, self.mu_m1to1)
        var_val = VarVal({self.k.name: 1, self.l.name: 0})
        self.assertEqual(evaluate(e, env_or_var_val=var_val), 0)

        # \int_x delta(-x - 1) H(x)
        # y = -x - 1, dy = - dx
        # = -\int_y delta(y) [y < -1] = 0
        delta = Delta(self.flip_shiftk)
        heaviside = Heaviside(self.shiftl)
        e = Int(delta * heaviside, self.mu_m1to1)
        var_val = VarVal({self.k.name: -1, self.l.name: 0})
        self.assertEqual(evaluate(e, env_or_var_val=var_val), 0)

        # \int_x delta(-x - 1) H(x)
        # y = -x - 1, dy = - dx
        # = -\int_y delta(y) H(-y - 1) = -\int_y delta(y) [y < -1]
        delta = Delta(self.flip_shiftk)
        heaviside = Heaviside(self.shiftl)
        e = Int(delta * heaviside, self.mu_m1to1)
        var_val = VarVal({self.k.name: -1, self.l.name: 0})
        self.assertEqual(evaluate(e, env_or_var_val=var_val), 0)

        # \int_x delta(-x - 1) H(x + 2)
        # y = -x - 1, dy = - dx
        # = -\int_y delta(y) [-y - 1 + 2 > 0] = 0
        # = -\int_y delta(y) [y < 1] = -1
        delta = Delta(self.flip_shiftk)
        heaviside = Heaviside(self.shiftl)
        mu = BoundedLebesgue(-2, 2, self.tvx)
        e = Int(delta * heaviside, mu)
        var_val = VarVal({self.k.name: -1, self.l.name: 1.5})
        self.assertEqual(evaluate(e, env_or_var_val=var_val), 1)

    @unittest.skip("Need to update malformed input checks")
    def test_delta_product_should_fail(self):
        delta = Delta(self.shiftk)
        with self.assertRaises(ValueError):
            expr = Int(delta * delta, self.mu_m1to1)
            var_val = VarVal({self.k.name: -1})
            e = evaluate_all(expr, env_or_var_val=var_val)

    def test_delta_alone_should_fail(self):
        delta = Delta(self.shiftk)
        # TODO: Maybe should be caught as ValueError?
        with self.assertRaises(KeyError):
            var_val = VarVal({self.k.name: -1})
            e = evaluate(delta, env_or_var_val=var_val)

    def test_delta_sums(self):
        delta1 = Delta(self.shiftk)
        delta = Delta(self.shiftl)
        mu = BoundedLebesgue(-2, 2, self.tvx)

        # \int_x delta(x + 1)x + delta(x)x = -1
        e = Int(delta1 * self.tvx + delta * self.tvx, mu)
        var_val = VarVal({self.k.name: 1, self.l.name: 0})
        self.assertEqual(evaluate(e, env_or_var_val=var_val), -1)

        # \int_x delta(x + 1)x + delta(x) (x + 1) = -1 + 1
        e = Int(delta1 * self.tvx + delta * (self.tvx + 1), mu)
        self.assertEqual(evaluate(e, env_or_var_val=var_val), 0)

        # TODO: non-uniform measure and assert that deltas without integrals blow up.

    def test_delta_heaviside_sum(self):
        # \int_x delta(x + 1) H(x + 2) + delta(x) H(x + 1) = 2

        delta1 = Delta(self.shiftk)
        heaviside2 = Heaviside(self.shiftm)
        delta = Delta(self.shiftl)
        heaviside1 = Heaviside(self.shiftk)

        e = Int(delta1 * heaviside2 + delta * heaviside1, BoundedLebesgue(-3, 3, self.tvx))
        var_val = VarVal({self.k.name: 1, self.l.name: 0, self.m.name: 2})
        self.assertAlmostEqual(evaluate_all(e, var_val, num_samples=50), 2)

    def test_delta_sum_with_added_variable(self):
        delta1 = Delta(self.shiftk)
        heaviside2 = Heaviside(self.shiftm)
        delta = Delta(self.shiftl)
        heaviside1 = Heaviside(self.shiftk)
        mu = BoundedLebesgue(-2, 1, self.tvx)
        # Sampling measures separately
        # delta(x + 1) H(x + 2) + delta(x) H(x + 1) + x
        # 1 + 1 + (x^2/2)|_-2^1 = 2 + (-1.5) = 0.5
        var_val = VarVal({self.k.name: 1, self.l.name: 0, self.m.name: 2})

        e = Int(delta * heaviside1, mu) / 10
        val = evaluate_all(e, env_or_var_val=var_val, num_samples=400)
        self.assertAlmostEqual(val, 0.1, places=2)

        e = Int(delta * heaviside1 + self.tvx, mu) / 10
        val = evaluate_all(e, env_or_var_val=var_val, num_samples=2000)
        self.assertAlmostEqual(val, -0.05, places=1)

        e = Int(delta1 * heaviside2 + self.tvx, mu) / 10
        val = evaluate_all(e, env_or_var_val=var_val, num_samples=1500)
        self.assertAlmostEqual(val, -0.05, places=1)

        e = Int(delta1 * heaviside2 + delta * heaviside1 + self.tvx, mu) / 10
        self.assertAlmostEqual(evaluate_all(e, var_val, num_samples=20000), 0.05, places=2)

    def test_different_variable_delta_and_heaviside(self):
        ym1 = Shift((self.l,), (self.tvy,))  # 1
        delta = Delta(self.shiftk)
        heaviside = Heaviside(ym1)

        # \int_x \int_y delta(x) H(y) = (\int_x delta(x)) (\int_y  H(y))
        # Since y is uniform(0, 1), Heaviside(identity_y) will always be 1
        nu = BoundedLebesgue(-1, 1, self.tvy)
        e = Int(Int(delta * heaviside, self.mu_m1to1), nu)
        var_val = VarVal({self.k.name: 0, self.l.name: 0})
        val = evaluate(e, var_val, 5000)
        self.assertAlmostEqual(1, val, 1)

    def test_product_times_measures(self):
        delta = Delta(self.shiftk)
        delta1 = Delta(self.shiftl)

        # \int_x (delta(x) + delta(x + 1)) x = (\int_x delta(x) x) + (\int_x delta(x + 1) x)
        mu = BoundedLebesgue(-2, 2, self.tvx)
        var_val = VarVal({self.k.name: 0, self.l.name: 1})
        e = Int(self.tvx * (delta + delta1), mu)
        self.assertEqual(evaluate(e, var_val), -1)


class TestIntegral(TestCase):
    def setUp(self):
        self.k0 = Var("k")
        self.k1 = Var("k")
        self.tvx = TegVar("x")
        self.tvy = TegVar("y")
        self.mu = BoundedLebesgue(0, 1, self.tvx)

    def test_integral_x(self):
        x = self.tvx
        self.assertAlmostEqual(evaluate(Int(x, self.mu), num_samples=1000), 0.5, places=1)
        self.assertAlmostEqual(evaluate(Int(x + 1, self.mu), num_samples=1000), 1.5, places=1)
        self.assertAlmostEqual(evaluate(Int(x * x, self.mu), num_samples=1000), 1 / 3, places=1)

        mu = BoundedLebesgue(-1.1, 2.4, self.tvx)
        self.assertAlmostEqual(evaluate(Int(x, mu), num_samples=50000), 2.275, places=1)
        self.assertAlmostEqual(evaluate(Int(x + 1, mu), num_samples=40000), 5.775, places=1)
        self.assertAlmostEqual(evaluate(Int(x * x, mu), num_samples=100000), 5.05167, places=1)

        # mu = BoundedLebesgue(-2, 1, self.tvx)
        # # Sampling measures separately
        # # x
        # # (x^2/2)|_-2^1 = -1.5
        # self.assertAlmostEqual(evaluate(Int(self.tvx, mu), num_samples=40000), -1.5, 1)

    def test_integral_sum_same_variable(self):
        x = self.tvx
        mu = BoundedLebesgue(Const(0), Const(1), x)
        nu = BoundedLebesgue(Const(-1), Const(0), x)
        # int_0^1 x^2 dx + int_-1^0 x dx = 1 / 3 - 1 / 2 = -1 / 6
        expr = Int(x**2, mu) + Int(x, nu)
        self.assertAlmostEqual(
            evaluate(expr, num_samples=1000),
            -1 / 6,
            places=1,
        )

    def test_integral_flipshift_heaviside(self):
        x = self.tvx
        a = Var("a")
        mu = BoundedLebesgue(Const(-2), Const(2), x)
        var_val = VarVal({a.name: 1})
        # TODO: check all Const(-1)
        # int_-2^2 H(1 - x) = 3
        expr = Int(Heaviside(FlipShift((a,), (x,))), mu)
        self.assertAlmostEqual(evaluate(expr, env_or_var_val=var_val, num_samples=5000), 3, places=1)

    def test_integral_division(self):
        x = self.tvx
        mu = BoundedLebesgue(Const(0), Const(1), x)
        # (1/3) / (1/2) = 2/3
        expr = Int(x**2, mu) / Int(x, mu)
        self.assertAlmostEqual(
            evaluate(expr, num_samples=2000),
            2 / 3,
            places=1,
        )

    @unittest.skip("Implement capture avoiding substitution")
    def test_integral_dx_dx(self):
        # TODO: Implement abstraction so that integrals don't capture variables
        x = self.tvx
        mu = BoundedLebesgue(Const(0), Const(1), x)
        nu = BoundedLebesgue(Const(-1), Const(0), x)
        # int_-1^0 x * int_0^1 x
        expr = Int(x * Int(x, mu), nu)
        self.assertAlmostEqual(evaluate(expr, num_samples=200), -0.25, places=1)

    def test_variable_of_integration_in_bounds_of_integration(self):
        k = Var("k")
        dk = Var("dk")
        x = TegVar("x")
        y = TegVar("y")
        mu = BoundedLebesgue(Const(0), y, x)
        nu = BoundedLebesgue(Const(0), Const(1), y)

        # d/dk int_0^1 (int_0^y k dx) dy
        e = Int(Int(k, mu), nu)

        # int_0^1 y dk dy
        # dk / 2
        dexpr = deriv(e, {k.name: dk.name})
        var_val = VarVal({k.name: 1, dk.name: 1})

        integral = evaluate(dexpr, env_or_var_val=var_val, num_samples=400)
        self.assertAlmostEqual(integral, 0.5, places=1)


class TestSquare(TestCase):
    def setUp(self):
        self.k = Var("k")
        self.l = Var("l")
        self.tvx = TegVar("x")
        self.tvy = TegVar("y")
        self.mu = BoundedLebesgue(0, 1, self.tvx)
        self.nu = BoundedLebesgue(0, 1, self.tvy)
        self.slice = SquareSlice((self.k,), (self.tvx, self.tvy), Const(1 / 2))
        self.off_axis_slice = OffAxisSquareSlice((self.k,), (self.tvx, self.tvy), Const(1 / 5))

    # @unittest.skip('Implement multivariate diffeomorphisms')
    def test_sliced_square(self):
        delta = Delta(self.slice)
        e = Int(Int(delta, self.mu), self.nu)
        var_val = VarVal({self.k.name: 0})
        self.assertAlmostEqual(evaluate(e, var_val, num_samples=40), 1 / 2, places=2)

    def test_sliced_square2(self):
        e = Int(Int(Heaviside(self.slice), self.mu), self.nu)
        dk = TegVar("dk")
        deriv_ctx = {self.k.name: dk.name}
        dintegral = deriv(e, deriv_ctx)
        var_val = VarVal({self.k.name: 0.5, dk.name: 1})
        self.assertAlmostEqual(evaluate(dintegral, num_samples=40, env_or_var_val=var_val), 1 / 2, places=2)

    def test_off_axis_sliced_square(self):
        delta = Delta(self.off_axis_slice)
        e = Int(Int(delta, self.mu), self.nu)
        var_val = VarVal({self.k.name: 0.5})
        self.assertAlmostEqual(evaluate(e, var_val, num_samples=40), 1 / 2, places=2)


class TestQuarterCircle(TestCase):
    def setUp(self):
        self.tvx = TegVar("x")
        self.mu = BoundedLebesgue(0, 1, self.tvx)
        self.tvy = TegVar("y")
        self.nu = BoundedLebesgue(0, 0.8, self.tvy)

        self.r = Var("r")
        self.quarter_circle = QuarterCircle((self.r,), (self.tvx, self.tvy))

    def test_mock(self):
        # \int_y 0.5(1 - y^2)^-0.5 dy
        var_val = VarVal({self.r.name: 1})
        expr = Int(Const(1 / 2) / Sqrt(self.r - self.tvy**2), self.nu)
        integ = evaluate(expr, env_or_var_val=var_val, num_samples=100)
        self.assertAlmostEqual(integ, 0.4636, places=1)

    def test_quarter_circle(self):
        # \int_x \int_y [0 < x < 1] [0 < y < 1] Heaviside(r - x^2 - y^2) / 2x
        heaviside = Heaviside(self.quarter_circle)
        nu = BoundedLebesgue(0, 1, self.tvy)
        e = Int(Int(heaviside, self.mu), nu)
        drsym = Sym("dr")
        deriv_ctx = {self.r.name: drsym}
        dintegral = deriv(e, deriv_ctx)
        var_val = VarVal({self.r.name: 1, drsym: 1})
        e = evaluate(dintegral, var_val, num_samples=5000)

        self.assertAlmostEqual(e, np.pi / 4, places=1)


class TestLinePlusCircle(TestCase):
    def setUp(self):
        self.tvx = TegVar("x")
        self.tvy = TegVar("y")
        self.mu = BoundedLebesgue(0, 1, self.tvx)
        self.nu = BoundedLebesgue(0, 1, self.tvy)
        self.t = Var("t")
        self.slice = SquareSlice((self.t,), (self.tvx, self.tvy), Const(1 / 2))
        self.r = Var("r")
        self.quarter_circle = QuarterCircle(
            (self.r,), (self.tvx, self.tvy), 1 / (2 * Sqrt(self.r - self.tvy * self.tvy))
        )

    def test_line_plus_circle(self):
        dirac_line = Delta(self.slice)
        dirac_circle = Delta(self.quarter_circle)
        var_val = VarVal({self.r.name: 1, self.t.name: 0})

        e = Int(Int(dirac_line, self.mu), self.nu)
        self.assertAlmostEqual(evaluate(e, env_or_var_val=var_val, num_samples=700), 1 / 2, places=1)

        e = Int(Int(dirac_circle, self.mu), self.nu)
        self.assertAlmostEqual(
            evaluate(e, env_or_var_val=var_val, num_samples=5000), np.pi * var_val[self.r.name] / 4, places=1
        )

        e = Int(Int(dirac_line + dirac_circle, self.mu), self.nu)
        self.assertAlmostEqual(
            evaluate(e, env_or_var_val=var_val, num_samples=5000), np.pi * var_val[self.r.name] / 4 + 1 / 2, places=1
        )

    def test_line_different_bounds(self):
        dirac_line = Delta(self.slice)

        mu = BoundedLebesgue(-1, 1, self.tvx)
        nu = BoundedLebesgue(-1, 1, self.tvy)
        e = Int(Int(dirac_line, mu), nu)
        var_val = VarVal({self.t.name: 0})
        self.assertAlmostEqual(evaluate(e, env_or_var_val=var_val, num_samples=700), 1, places=1)

        var_val = VarVal({self.t.name: 2})
        self.assertAlmostEqual(evaluate(e, env_or_var_val=var_val, num_samples=700), 0.5, places=1)

        mu = BoundedLebesgue(-2, 2, self.tvx)
        nu = BoundedLebesgue(-1, 1, self.tvy)
        e = Int(Int(dirac_line, mu), nu)
        var_val = VarVal({self.t.name: 2})
        self.assertAlmostEqual(evaluate(e, env_or_var_val=var_val, num_samples=700), 1, places=1)

        mu = BoundedLebesgue(-2, 2, self.tvx)
        nu = BoundedLebesgue(-2, 2, self.tvy)
        e = Int(Int(dirac_line, mu), nu)
        var_val = VarVal({self.t.name: 2})
        self.assertAlmostEqual(evaluate(e, env_or_var_val=var_val, num_samples=700), 2, places=1)


class TestSimplex3D(TestCase):
    def setUp(self):
        self.tvx = TegVar("x")
        self.tvy = TegVar("y")
        self.tvz = TegVar("z")
        self.mu = BoundedLebesgue(-9999, 9999, self.tvx)
        self.mu01 = BoundedLebesgue(0, 1, self.tvx)
        self.nu = BoundedLebesgue(0, 1, self.tvy)
        self.ga = BoundedLebesgue(0, 1, self.tvz)
        self.simplex3d = Simplex3D(tuple(), (self.tvx, self.tvy, self.tvz))

    def test_simplex(self):
        dirac = Delta(self.simplex3d)
        var_val = VarVal()

        # int_0^1 int_0^1 int_-9999^9999 delta(1 - x - y - z) dx dy dz
        e = Int(Int(Int(dirac, self.mu), self.nu), self.ga)
        simplex_area = evaluate(e, var_val, 1)
        self.assertAlmostEqual(simplex_area, 1, places=1)

        # int_0^1 int_0^1 int_0^1 delta(1 - x - y - z) dx dy dz
        e = Int(Int(Int(dirac, self.mu01), self.nu), self.ga)
        simplex_area = evaluate(e, var_val, 500)
        self.assertAlmostEqual(simplex_area, 1 / 2, places=1)


class TestAbstraction(TestCase):
    def setUp(self):
        def potto_fix(f: Function):
            ret_args = f.arg_names[1:]
            ret_body = App(f, (f, *ret_args))
            ret_name = Sym(f"fix_{f.name.name}")
            return Function(ret_args, ret_body, ret_name)

        self.potto_fix = potto_fix

    def test_delta_in_abstraction(self):

        a0 = Var("a0")
        a1 = TegVar("a1")
        fbody = Heaviside(Shift((a0,), (a1,)))
        f = Function((a0, a1), fbody, Sym("f"))

        a = Var("a")
        x = TegVar("x")
        measure = BoundedLebesgue(0, 1, x)
        integral = Int(App(f, (a, x), "f_app"), measure)

        dasym = Sym("da")
        deriv_ctx = {a.name: dasym}
        dintegral = deriv(integral, deriv_ctx)

        var_val = VarVal({a.name: -0.5, dasym: 0})
        val = evaluate_all(dintegral, env_or_var_val=var_val, num_samples=1)
        self.assertEqual(val, 0)

        var_val = VarVal({a.name: -0.5, dasym: 1})
        val = evaluate_all(dintegral, env_or_var_val=var_val)
        self.assertEqual(val, 1)

    def test_delta_in_application(self):

        a0 = Var("a0")
        a1 = TegVar("a1")
        f = Function((a0, a1), a0 * a1, Sym("f"))

        a = Var("a")
        x = TegVar("x")
        heaviside = Heaviside(Shift((a,), (x,)))
        measure = BoundedLebesgue(-1, 1, x)
        integral = Int(App(f, (heaviside, x), "f_app"), measure)
        # int (\p, q p*q) (H(a + x), x)
        # int (\p, q, dp, dq. p*dq + q*dp) (H(a + x), x, delta(a + x)da, 0)
        # int H(a + x)*0 + x*delta(a + x)da = -a da

        dasym = Sym("da")
        deriv_ctx = {a.name: dasym}
        dintegral = deriv(integral, deriv_ctx)

        var_val = VarVal({a.name: -0.5, dasym: 0})
        samples = generate(to_irenv(dintegral), to_env(var_val), 1, TraceEnv())
        traces = extract_traces_from_name(samples, x.name, with_arg_nums=True)
        expected_trace_names = (
            TraceName.BinopLeft,
            TraceName.BinopLeft,
            TraceName.Integral,
            TraceName.AppArg,
            TraceName.BinopRight,
            TraceName.BinopLeft,
            TraceName.BinopLeft,
            TraceName.AppFun,
            TraceName.Leaf,
        )
        expected_arg_nums = (None, None, None, 2, None, None, None, None, None)
        self.assertEqual(traces, {tuple(zip(expected_trace_names, expected_arg_nums))})
        val = evaluate_all(dintegral, var_val)
        self.assertEqual(val, 0)

        var_val = VarVal({a.name: -0.5, dasym: 1})
        val = evaluate_all(dintegral, var_val, 1)
        self.assertEqual(val, 0.5)

    def test_deriv_heaviside(self):
        x = TegVar("x")
        t = Var("t")
        integrand = Heaviside(Shift((t,), (x,)))
        mu = BoundedLebesgue(-1, 1, x)
        result = Int(integrand, mu)
        dt = Sym("dt")
        result1 = deriv(result, {t.name: dt})
        self.assertEqual(1, evaluate_all(result1, env_or_var_val=VarVal({t.name: 0.5, dt: 1})))

    def test_function_arg_in_app(self):
        f0 = Var("f0")
        x0 = Var("x0")
        # f0 is an abstract variable representing a function
        # but app takes in a function type
        # lambda f0, x0. f0(x0)
        apply_func = Function((f0, x0), App(f0, (x0,)), Sym("apply"))

        a = Var("a")
        # ((lambda f0, x0. f0(x0)) (lambda x0. x0, a)) = ((lambda x0. x0) a) = a
        identity = App(apply_func, (Function((x0,), x0), a))

        var_val = VarVal({a.name: 3})
        self.assertEqual(evaluate_all(identity, var_val), 3)

        var_val = VarVal({a.name: -2.123})
        self.assertEqual(evaluate_all(identity, var_val), -2.123)

        da = Var("da")
        id_on_da = deriv(identity, {a.name: da.name})

        var_val = VarVal({a.name: -2.123, da.name: 0.1})
        self.assertEqual(evaluate_all(id_on_da, var_val), 0.1)

        var_val = VarVal({a.name: 5, da.name: -10})
        self.assertEqual(evaluate_all(id_on_da, var_val), -10)

        # Int(App(apply_func, (Function((x0,), x0), Heaviside(a)))

        # def pyf(x_):
        #     return x_ * 2

        # x1 = Var("x1")
        # f = Function((x1,), pyf(x1), Sym("f"))
        # x = Var("x")
        # expr = App(apply_func, (f, x))

        # var_val = VarVal({x.name: 3})
        # val = evaluate_all(expr, env_or_var_val=var_val)
        # print(val)

    def test_factorial(self):
        def py_fix(f):
            def f2(*args):
                return f(f, *args)

            return f2

        def py_fact(f, n):
            if n == 0:
                return 1
            else:
                return n * f(f, n - 1)

        n0 = Var("n0")
        f0 = Var("f0")
        neg_one_half = Var("bad")

        # if c <= 0.5 then b else a
        #
        # H(c - 0.5) * a + (1-H(c - 0.5)) * b
        #
        # b + H(c - 0.5) * (a-b)

        fact_recurse = n0 * App(f0, (f0, n0 - 1))
        fact_default = Const(1)
        fact_body = IfElse(n0, fact_recurse, fact_default)
        fact_func = Function((f0, n0), fact_body, Sym("fact"))

        fix_fact = self.potto_fix(fact_func)

        n = Var("n")
        expr = App(fix_fact, (n,))
        var_val = VarVal({n.name: 6, neg_one_half.name: -1 / 2})
        val = evaluate_all(expr, env_or_var_val=var_val)
        self.assertEqual(720, val)

    def test_delta_in_recursive_base_case(self):
        n0 = Var("n0")
        f0 = Var("f0")

        t = Var("t")
        x = TegVar("x")

        fact_recurse = n0 + App(f0, (f0, n0 - 1))
        fact_default = x * Heaviside(FlipShift((t,), (x,)))
        fact_body = IfElse(n0, fact_recurse, fact_default)
        fact_func = Function((f0, n0), fact_body, Sym("fact"))

        fix_fact = self.potto_fix(fact_func)

        n = Var("n")
        integrand = App(fix_fact, (n,))

        mu = BoundedLebesgue(Const(0), Const(1), x)
        expr = Int(integrand, mu)

        var_val = VarVal({n.name: 6, t.name: 0.7})
        val = evaluate_all(expr, env_or_var_val=var_val, num_samples=1000)
        self.assertAlmostEqual(7 * 6 // 2 + 0.5 * (0.75**2), val, places=1)

        dt = TegVar("dt")
        dn = TegVar("dn")
        dexpr = deriv(expr, {t.name: dt.name, n.name: dn.name})

        var_val = VarVal({n.name: 6, dn.name: 0, t.name: 0.7, dt.name: 1})
        dval = evaluate_all(dexpr, env_or_var_val=var_val)
        self.assertAlmostEqual(0.7, dval, places=2)

    def test_delta_comb_recursive(self):
        n0 = Var("n0")
        f0 = Var("f0")

        t = Var("t")
        x = TegVar("x")

        t_ = Var("t_")
        comb_fiber = App(Function((t_,), x / 10 * Heaviside(FlipShift((t_,), (x,)))), (t + n0,))
        fact_recurse = comb_fiber + App(f0, (f0, n0 - 1))
        fact_default = Const(0)
        fact_body = IfElse(n0, fact_recurse, fact_default)
        fact_func = Function((f0, n0), fact_body, Sym("fact"))
        """
        the above in python:
        x = ...
        def pyf(n):
            if n > 0:
                return x*H(t+n-x) + pyf(n-1)
            else:
                return x * x
        
        the above in math:
        f(t) \int_x x^2 + \sum_{k=1}^n x*H(t+k-x)
        d/dt f(t) =  \int_x \sum_{k=1}^n x*delta(t+k-x)
        n = 1 0.15
        n = 2 0.3
        """

        fix_fact = self.potto_fix(fact_func)

        n = Var("n")
        integrand = App(fix_fact, (n,))

        mu = BoundedLebesgue(Const(0), Const(10), x)
        expr = Int(integrand, mu)

        var_val = VarVal({n.name: 6, t.name: 0.5})
        val = evaluate_all(expr, env_or_var_val=var_val, num_samples=15000)
        self.assertAlmostEqual(5.675, val, places=1)

        dt = TegVar("dt")
        dn = TegVar("dn")
        dexpr = deriv(expr, {t.name: dt.name, n.name: dn.name})

        var_val = VarVal({n.name: 6, dn.name: 0, t.name: 0.5, dt.name: 1})
        dval = evaluate_all(dexpr, env_or_var_val=var_val, num_samples=50)
        self.assertAlmostEqual(2.4, dval, places=2)

    @unittest.skip("Need to update test case with new fix api")
    def test_do_n_times(self):
        #
        # sum $ for i in range(10) seq:
        #     f(i)
        #
        # def reduce(s, f, default):
        #     match s:
        #         case []:
        #             return default
        #         case a, s:
        #             # return f(a, reduce(s, f, default))
        #             return reduce(s, f, f(default, a))
        # def sum(s):
        #     return reduce(s, lambda x, y: x + y, 0)
        #
        # for i in range(10) seq-reduce(r):
        #     f(i)
        #
        # r(f(0), r(f(1), r(f(2), f(3))))
        # f(f(f(f(x))))
        #
        # def seq_reduce(f, r, n):
        #     if n <= 0:
        #         return f(n)
        #     else:
        #         return r(f(n), seq_reduce(f, r, n-1))
        #
        # def seq_reduce(f, r, n, default):
        #     if n < -0.5:
        #         return default
        #     else:
        #         return seq_reduce(f, r, n-1, r(default, f(n)))
        #
        # if c <= 0 then b else a
        # if c > 0 then a else b
        #
        # H(c) * a + (1-H(c)) * b
        #
        # b + H(c) * (a-b)
        #
        # cond_lt_0 = n + 0.5

        f0 = Var("f0")
        r0 = Var("r0")
        n0 = Var("n0")
        vardefault = Var("default0")

        do_n_name = Sym("do_n_times")
        do_n_func = Function((f0.name, r0.name, n0.name, vardefault.name), None, do_n_name)
        # if n < 0.5:
        #   default
        # else:
        #   do_n(f, r, n-1, r(default, f(n)))
        if_clause = vardefault
        else_clause = App((f0, r0, n0 - 1, App((vardefault, App(n0, f0)), r0)), do_n_func)
        onehalf = Var("onehalf")
        do_n_body = if_clause + Heaviside(Shift((onehalf,), (n0,))) * (else_clause - if_clause)

        from dataclasses import replace

        do_n_func = replace(
            do_n_func, body=do_n_body, name=do_n_name
        )  # TODO: jesse switch this out w/ non-explicit recursion

        def pyf(i0_):
            return i0_ * i0_

        i0 = Var("i0")
        f_name = Sym("f")
        f_func = Function((i0,), pyf(i0), f_name)

        def pyreduce(d0_, v0_):
            return d0_ + v0_

        d0 = Var("d0")
        v0 = Var("v0")
        reduce_name = Sym("r")
        reduce_func = Function((d0, v0), pyreduce(d0, v0), reduce_name)

        n0 = Var("n0")
        default = Var("default_")
        expr = App((f_func, reduce_func, n0, default), do_n_func)

        var_val = VarVal({n0.name: 3, default: 0})

        val = evaluate_all(expr, env_or_var_val=var_val)
        # print(val)

        assert False


class TestLeibniz(TestCase):
    def setUp(self):
        self.k = Var("k")
        self.dk = Var("dk")
        self.tvx = TegVar("x")
        self.nu = BoundedLebesgue(Const(0), self.k, self.tvx)

    def test_variable_in_bounds_of_integration(self):
        # d/dk int_0^k 1 dx
        e = Int(Const(1), self.nu)

        # int_0^k 1 dx + dk
        # int_0^2 0 dx + 1 = 1
        dexpr = deriv(e, {self.k.name: self.dk.name})
        var_val = VarVal({self.k.name: 2, self.dk.name: 1})
        self.assertAlmostEqual(evaluate(dexpr, env_or_var_val=var_val, num_samples=10), 1, places=2)

        # d/dk int_0^k  k x  dx
        e = Int(self.k * self.tvx, self.nu)

        # int_0^k  dk x  dx + k^2 dk
        # int_0^2 x dx + 4 = x^2 / 2 |_0^2 + 4 = 6
        dexpr = deriv(e, {self.k.name: self.dk.name})
        var_val = VarVal({self.k.name: 2, self.dk.name: 1})
        self.assertAlmostEqual(evaluate(dexpr, env_or_var_val=var_val, num_samples=10000), 6, places=1)


class TestNonBoundedLebesgueMeasure(TestCase):
    def test_uniform_measure(self):
        # int_0^1 1 dx
        x = TegVar("x")
        nu = Uniform(0, 1, x)
        e = Int(Const(1), nu)

        # int_0^1 1 duniform(0, 1)
        # int_0^1 1 / 1 dx = 1
        self.assertAlmostEqual(evaluate(e, num_samples=10), 1, places=2)

        mu = Uniform(-1.2, 2.34, x)
        e = Int(Const(1), mu)
        self.assertAlmostEqual(evaluate(e, num_samples=10), 1, places=2)

    def test_triangle_measure(self):
        # int_0^1 1 dx
        x = TegVar("x")
        nu = Triangular(-1, 1, x)
        e = Int(Const(1), nu)

        # int_0^1 1 duniform(0, 1)
        # int_0^1 1 / 1 dx = 1
        self.assertAlmostEqual(evaluate(e, num_samples=5000), 1, places=2)


class TestFunctionalBehavior(TestCase):
    def test_square(self):
        x = Var("x")
        expr = x * x
        squ = Function((x,), expr)
        self.assertAlmostEqual(evaluate(App(squ, (Const(20),))), 400)

    def test_square_higher_order(self):
        x = Var("x")
        y = Var("y")
        expr_1 = x * x
        expr_2 = y
        squ = Function((x,), expr_1)
        out = Function((y,), expr_2)
        self.assertAlmostEqual(evaluate(App(App(out, (squ,)), (Const(20),))), 400)


class TestDiffeomorphicIfElse(TestCase):
    class Scale(Affine):
        def function(self, vars: tuple[GExpr, ...], tvars: tuple[GExpr, ...]) -> tuple[GExpr, ...]:
            return tuple(tvar * var for var, tvar in zip(vars, tvars))

        def inverse(self, vars: tuple[GExpr, ...], tvars: tuple[GExpr, ...]) -> tuple[GExpr, ...]:
            return tuple(tvar / var for var, tvar in zip(vars, tvars))

    def scale(vars, tvars):
        return TestDiffeomorphicIfElse.Scale(vars, tvars)

    class Identity(Affine):
        def function(self, vars: tuple[GExpr, ...], tvars: tuple[GExpr, ...]) -> tuple[GExpr, ...]:
            return tvars

        def inverse(self, vars: tuple[GExpr, ...], tvars: tuple[GExpr, ...]) -> tuple[GExpr, ...]:
            return tvars

    def identity(tvars):
        return TestDiffeomorphicIfElse.Identity(tuple(), tvars)

    def test_ifelse_composed_diffeos(self):
        x = TegVar("x")
        k = Var("k")
        l = Var("l")
        m = Var("m")
        y = TegVar("y")

        nu = BoundedLebesgue(0, 1, x)
        z = TegVar("z")
        dk = Var("dk")
        dl = Var("dl")
        dm = Var("dm")

        scaled_x = TestDiffeomorphicIfElse.scale((m,), (x,))

        # e = \int^0_1 (m*x + k > 0) ? (2l+x) : (3x)
        f_inner = IfElse(TestDiffeomorphicIfElse.identity((z,)), l * 2 + x, x * 3)
        f_ifelse = Function((z,), f_inner)
        shift_y = Shift((k,), (y,))
        integrand = App(Function((y,), App(f_ifelse, (shift_y,))), (scaled_x,))
        expr = Int(integrand, nu)
        dexpr = deriv(expr, {k.name: dk.name, l.name: dl.name, m.name: dm.name})

        bindings = VarVal({k.name: -0.5, l.name: 4.0, m.name: 1.0, dk.name: 1.0, dl.name: 0.0, dm.name: 0.0})
        self.assertAlmostEqual(evaluate_all(dexpr, bindings, num_samples=1000), 7.0, places=1)

        bindings = VarVal({k.name: -1.5, l.name: 4.0, m.name: 1.0, dk.name: 1.0, dl.name: 0.0, dm.name: 0.0})
        self.assertAlmostEqual(evaluate_all(dexpr, bindings, num_samples=100), 0.0)

    def test_ifelse_single_integral(self):
        x = TegVar("x")
        k = Var("k")
        l = Var("l")

        shiftl = Shift((k,), (x,))
        ifelse = IfElse(shiftl, l * 2 + x, x * 3)
        nu = BoundedLebesgue(0, 1, x)

        # e = \int^0_1 (x + k > 0) ? (2l+x) : (3x)
        expr = Int(ifelse, nu)

        dk = Var("dk")
        dl = Var("dl")
        dexpr = deriv(expr, {k.name: dk.name, l.name: dl.name})

        bindings = VarVal({k.name: -0.5, l.name: 4.0, dk.name: 1.0, dl.name: 0.0})
        self.assertAlmostEqual(evaluate(dexpr, bindings, num_samples=100), 7.0)

        bindings = VarVal({k.name: -1.5, l.name: 4.0, dk.name: 1.0, dl.name: 0.0})
        self.assertAlmostEqual(evaluate(dexpr, bindings, num_samples=100), 0.0)


class TestSanity(TestCase):
    def test_div(self):
        x = TegVar("x")
        k, dk = Var("k"), Var("dk")
        l, dl = Var("l"), Var("dl")

        x = (k + l) / (k + l)
        y = Const(1)
        dx = deriv(x, {k.name: dk.name, l.name: dl.name})
        dy = deriv(y, {k.name: dk.name, l.name: dl.name})

        bindings = VarVal({k.name: -0.5, l.name: 4.0, dk.name: 1.0, dl.name: 0.0})
        self.assertAlmostEqual(evaluate(x, bindings, num_samples=100), evaluate(y, bindings, num_samples=100))
        self.assertAlmostEqual(evaluate(dx, bindings, num_samples=100), evaluate(dy, bindings, num_samples=100))

    def test_app_of_tvars_not_diffeod_not_in_if(self):
        x, y = TegVar("x"), TegVar("y")
        k, dk = Var("k"), Var("dk")
        l, dl = Var("l"), Var("dl")
        mu = BoundedLebesgue(0, 1, x)
        nu = BoundedLebesgue(0, 1, y)
        alpha = 2 * x * y
        beta = 2 * x + 2 * y

        # int int 4 * x * y * l + 6 * x + 6 * y * k dx dy
        a, b = Var("a"), Var("b")
        shader = Function((a, b), (2 * a * l + 3 * b * k) / 10)
        integrand = App(shader, (alpha, beta))

        x = Int(Int(integrand, mu), nu)
        dx = deriv(x, {k.name: dk.name, l.name: dl.name})

        bindings = VarVal({k.name: 1, l.name: 1, dk.name: 1.0, dl.name: 0.0})
        self.assertAlmostEqual(evaluate(x, bindings, num_samples=1000), 0.7, places=1)
        self.assertAlmostEqual(evaluate(dx, bindings, num_samples=1000), 0.6, places=1)


def affine(x, y, a, b, c):
    # a_, b_, c_ = Var("a"), Var("b"), Var("c")
    return Affine((a, b, c), (x, y))
    # return Affine2((a, b, c), (x, y))


def half_space(x, y, a, b, c):
    tv0, tv1 = TegVar("tv0"), TegVar("tv1")
    a_, b_, c_ = Var("a_"), Var("b_"), Var("c_")
    return App(
        Function(
            (a_, b_, c_), App(Function((tv0, tv1), IfElse(tv0, Const(0.0), Const(1.0))), (affine(x, y, a_, b_, c_),))
        ),
        (a, b, c),
    )


class TestVaryingInteriorDelta(TestCase):
    def setUp(self):
        self.x, self.y = TegVar("x"), TegVar("y")
        self.a, self.b, self.c = Var("a"), Var("b"), Var("c")
        self.da, self.db, self.dc = Var("da"), Var("db"), Var("dc")
        self.h = half_space(self.x, self.y, self.a, self.b, self.c)

    def test_const_interior(self):
        expr = Int(Int(self.h, BoundedLebesgue(0, 1, self.x)), BoundedLebesgue(0, 1, self.y))
        dexpr = deriv(expr, {self.a.name: self.da.name, self.b.name: self.db.name, self.c.name: self.dc.name})
        # Expect: -1 (works)
        result = evaluate(
            dexpr,
            VarVal(
                {
                    self.a.name: 1.0,
                    self.b.name: -1.0,
                    self.c.name: 0.0,
                    self.da.name: 0.0,
                    self.db.name: 0.0,
                    self.dc.name: 1.0,
                }
            ),
            num_samples=1000,
        )
        self.assertAlmostEqual(result, -1.0, places=1)

    def test_linear_interior(self):
        expr = Int(Int(self.h * self.x, BoundedLebesgue(0, 1, self.x)), BoundedLebesgue(0, 1, self.y))
        dexpr = deriv(expr, {self.a.name: self.da.name, self.b.name: self.db.name, self.c.name: self.dc.name})
        # Expect: -0.5 (works)
        result = evaluate(
            dexpr,
            VarVal(
                {
                    self.a.name: 1.0,
                    self.b.name: -1.0,
                    self.c.name: 0.0,
                    self.da.name: 0.0,
                    self.db.name: 0.0,
                    self.dc.name: 1.0,
                }
            ),
            num_samples=1000,
        )
        self.assertAlmostEqual(result, -0.5, places=1)

    def test_quadratic_interior(self):
        expr = Int(
            Int(self.h * (4 * self.x * (1 - self.x)), BoundedLebesgue(0, 1, self.x)), BoundedLebesgue(0, 1, self.y)
        )
        dexpr = deriv(expr, {self.a.name: self.da.name, self.b.name: self.db.name, self.c.name: self.dc.name})
        # Expect: -0.6667 (results in -0.833)
        result = evaluate(
            dexpr,
            VarVal(
                {
                    self.a.name: 1.0,
                    self.b.name: -1.0,
                    self.c.name: 0.0,
                    self.da.name: 0.0,
                    self.db.name: 0.0,
                    self.dc.name: 1.0,
                }
            ),
            num_samples=1000,
        )
        self.assertAlmostEqual(result, -0.6667, places=1)


class TestGuardedDiffeomorphism(TestCase):
    class SimpleSqrtDiffeo(Diffeomorphism):
        def __str__(self):
            return f"SQRT[{self.tvars[0]} - {self.vars[0]}]"

        def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
            return (Sqrt(tvars[0] - vars[0]),)

        def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
            return ((tvars[0] + vars[0]) ** 2,)

    @unittest.skip("Need to catch divide by zero errors in square rots")
    def test_guarded_sqrt(self):
        rad = Var("r")
        x = TegVar("x")
        drad = Var("drad")
        infini_map = {rad.name: drad.name}
        vv_map_loss = VarVal({rad.name: 0, drad.name: 1})
        bnd = IfElse(
            ShiftRight((rad,), (x,), Const(1)), Heaviside(self.SimpleSqrtDiffeo((rad,), (x,), Const(1))), Const(0)
        )
        x_m = BoundedLebesgue(-10, 10, x)
        int_b = Int(bnd, x_m)
        deriv_b = deriv(int_b, infini_map)
        self.assertAlmostEqual(evaluate_all(deriv_b, env_or_var_val=vv_map_loss, num_samples=1000), 1.0, places=1)


def extract_traces_from_name(gen: Gen, name: Sym, with_arg_nums=False) -> set[tuple]:
    # variables = [i for i in gen.keys()]
    # trace = (samples for samples in gen[variables[1]])
    def iter_with_arg_num(t: Trace):
        if t.name is not None:
            yield (t.name, t.arg_num) if with_arg_nums else t.name
            if t.next_trace is not None:
                yield from (i for i in iter_with_arg_num(t.next_trace))

    return {tuple(i for i in iter_with_arg_num(sample.trace)) for sample in gen[name] if sample.trace is not None}


class TestTracedGenerate(TestCase):
    def setUp(self) -> None:
        self.gen = lambda expr, var_val: generate(to_irenv(expr), to_env(var_val), 1, TraceEnv())

    def test_delta(self):
        x = TegVar("x")
        mu = BoundedLebesgue(-1, 1, x)
        l = Var("l")
        var_val = VarVal({l.name: 0})
        shiftl = Shift((l,), (x,))

        expr = Int(Delta(shiftl), mu)
        gen = self.gen(expr, var_val)
        traces = extract_traces_from_name(gen, x.name)
        self.assertEqual(traces, {(TraceName.Integral, TraceName.AppFun, TraceName.Leaf)})
        val = evaluate(expr, var_val, 500)
        self.assertAlmostEqual(val, 1, 1)

        expr = Int(x + Delta(shiftl), mu)
        gen = self.gen(expr, var_val)
        traces = extract_traces_from_name(gen, x.name)
        self.assertEqual(traces, {(TraceName.Integral, TraceName.BinopRight, TraceName.AppFun, TraceName.Leaf)})
        val = evaluate(expr, var_val, 4000)
        self.assertAlmostEqual(val, 1, 1)

        expr = Int(Delta(shiftl) * x, mu)
        gen = self.gen(expr, var_val)
        traces = extract_traces_from_name(gen, x.name)
        self.assertEqual(traces, {(TraceName.Integral, TraceName.BinopLeft, TraceName.AppFun, TraceName.Leaf)})
        val = evaluate(expr, var_val, 1)
        self.assertEqual(val, 0)

        expr = Int(Delta(shiftl) / (x + 1), mu)
        gen = self.gen(expr, var_val)
        traces = extract_traces_from_name(gen, x.name)
        self.assertEqual(traces, {(TraceName.Integral, TraceName.BinopLeft, TraceName.AppFun, TraceName.Leaf)})
        val = evaluate(expr, var_val, 1)
        self.assertEqual(val, 1)

        gen = self.gen(Int(Delta(shiftl), mu), var_val)
        traces = extract_traces_from_name(gen, x.name)
        self.assertEqual(traces, {(TraceName.Integral, TraceName.AppFun, TraceName.Leaf)})
        val = evaluate(expr, var_val, 1)
        self.assertEqual(val, 1)

        expr = Int(Delta(shiftl) + Delta(shiftl), mu)
        gen = self.gen(expr, var_val)
        traces = extract_traces_from_name(gen, x.name)
        expected_traces = {
            (TraceName.Integral, TraceName.BinopLeft, TraceName.AppFun, TraceName.Leaf),
            (TraceName.Integral, TraceName.BinopRight, TraceName.AppFun, TraceName.Leaf),
        }
        self.assertEqual(traces, expected_traces)
        val = evaluate(expr, var_val, 1)
        self.assertEqual(val, 2)

    def test_if_else(self):
        x = TegVar("x")
        mu = BoundedLebesgue(-1, -0.5, x)
        l = Var("l")
        vv = VarVal({l.name: 0.75, x.name: 1})
        shiftl = Shift((l,), (x,))
        # [x > 0] Delta(shiftl) + [x < 0] x * Delta(shiftl)
        expr = Int(IfElse(x, Delta(shiftl), x * Delta(shiftl)), mu)
        gen = self.gen(expr, vv)
        traces = extract_traces_from_name(gen, x.name)
        expected_traces = {
            (TraceName.Integral, TraceName.ElseBody, TraceName.BinopRight, TraceName.AppFun, TraceName.Leaf),
            (TraceName.Integral, TraceName.IfBody, TraceName.AppFun, TraceName.Leaf),
        }
        self.assertEqual(traces, expected_traces)
        val = evaluate(expr, vv, 1)
        self.assertEqual(val, -0.75)

    def test_delta_comb(self):
        def potto_fix(f: Function):
            ret_args = f.arg_names[1:]
            ret_body = App(f, (f, *ret_args))
            ret_name = Sym(f"fix_{f.name.name}")
            return Function(ret_args, ret_body, ret_name)

        n0 = Var("n0")
        f0 = Var("f0")

        t = Var("t")
        x = TegVar("x")

        t_ = Var("t_")
        comb_fiber = App(Function((t_,), x / 10 * Heaviside(FlipShift((t_,), (x,)))), (t + n0,))
        fact_recurse = comb_fiber + App(f0, (f0, n0 - 1))
        fact_default = Const(0)
        fact_body = IfElse(n0, fact_recurse, fact_default)
        fact_func = Function((f0, n0), fact_body, Sym("fact"))

        fix_fact = potto_fix(fact_func)

        n = Var("n")
        integrand = App(fix_fact, (n,))

        mu = BoundedLebesgue(Const(0), Const(10), x)
        expr = Int(integrand, mu)

        var_val = VarVal({n.name: 6, t.name: 0.5})
        gen = self.gen(expr, var_val)
        traces = extract_traces_from_name(gen, x.name)
        self.assertEqual(traces, set())
        # val = evaluate_all(expr, env_or_var_val=var_val, num_samples=3000)

        dt = TegVar("dt")
        dn = TegVar("dn")
        dexpr = deriv(expr, {t.name: dt.name, n.name: dn.name})

        var_val = VarVal({n.name: 3, dn.name: 0, t.name: 0.5, dt.name: 1})
        gen = self.gen(dexpr, var_val)
        traces = extract_traces_from_name(gen, x.name)
        expected_traces = {
            (
                TraceName.BinopLeft,
                TraceName.BinopLeft,
                TraceName.Integral,
                TraceName.AppFun,
                TraceName.AppFun,
                TraceName.IfBody,
                TraceName.BinopRight,
                TraceName.AppFun,
                TraceName.IfBody,
                TraceName.BinopRight,
                TraceName.AppFun,
                TraceName.IfBody,
                TraceName.BinopLeft,
                TraceName.AppFun,
                TraceName.BinopRight,
                TraceName.BinopRight,
                TraceName.BinopRight,
                TraceName.BinopLeft,
                TraceName.BinopLeft,
                TraceName.AppFun,
                TraceName.Leaf,
            ),
            (
                TraceName.BinopLeft,
                TraceName.BinopLeft,
                TraceName.Integral,
                TraceName.AppFun,
                TraceName.AppFun,
                TraceName.IfBody,
                TraceName.BinopRight,
                TraceName.AppFun,
                TraceName.IfBody,
                TraceName.BinopLeft,
                TraceName.AppFun,
                TraceName.BinopRight,
                TraceName.BinopRight,
                TraceName.BinopRight,
                TraceName.BinopLeft,
                TraceName.BinopLeft,
                TraceName.AppFun,
                TraceName.Leaf,
            ),
            (
                TraceName.BinopLeft,
                TraceName.BinopLeft,
                TraceName.Integral,
                TraceName.AppFun,
                TraceName.AppFun,
                TraceName.IfBody,
                TraceName.BinopLeft,
                TraceName.AppFun,
                TraceName.BinopRight,
                TraceName.BinopRight,
                TraceName.BinopRight,
                TraceName.BinopLeft,
                TraceName.BinopLeft,
                TraceName.AppFun,
                TraceName.Leaf,
            ),
        }
        self.assertEqual(traces, expected_traces)
        var_val = VarVal({n.name: 6, dn.name: 0, t.name: 0.5, dt.name: 1})
        dval = evaluate(dexpr, var_val, 1)
        self.assertAlmostEqual(2.4, dval, places=2)

    def test_delta_in_arg_app(self):
        f0 = Var("f0")
        x0 = TegVar("x0")
        # f0 is an abstract variable representing a function
        # but app takes in a function type
        # lambda f0, x0. f0(delta(x0))
        x = TegVar("x")
        x1 = Var("x1")
        a = Var("a")
        diffeo = Shift((a,), (x,))

        mu = BoundedLebesgue(0, 1, x)

        # (lambda x1. x1)(delta(a + x))
        fun = Function((x1,), x1)
        expr = Int(App(fun, (Delta(diffeo),)), mu)
        var_val = VarVal({a.name: 0.5})
        traces = extract_traces_from_name(self.gen(expr, var_val), x.name)
        expected_traces = {(TraceName.Integral, TraceName.AppArg, TraceName.AppFun, TraceName.Leaf)}
        self.assertEqual(traces, expected_traces)
        self.assertEqual(evaluate(expr, var_val, 1), 0)

        var_val = VarVal({a.name: -0.5})
        expr = Int(App(fun, (Delta(diffeo),)), mu)
        self.assertEqual(evaluate(expr, var_val, 1), 1)

        # ((lambda f0, x0. f0(delta(a + x0))) (lambda x1. x1, x))
        # = ((lambda x1. x1) delta(a + x)) = delta(a + x)
        diffeo = Shift((a,), (x0,))
        apply_func = Function((f0, x0), App(f0, (Delta(diffeo),)))
        identity = App(apply_func, (Function((x1,), x1), x))
        expr = Int(identity, mu)
        var_val = VarVal({a.name: 3})
        gen = self.gen(expr, var_val)

        traces = extract_traces_from_name(gen, x.name)  # TODO: multiple None trace samples??? bug?
        expected_trace = {(TraceName.Integral, TraceName.AppFun, TraceName.AppArg, TraceName.AppFun, TraceName.Leaf)}
        self.assertEqual(traces, expected_trace)
        val = evaluate(expr, var_val, 1)
        self.assertEqual(val, 0)
        var_val = VarVal({a.name: -0.5})
        val = evaluate(expr, var_val, 1)
        self.assertEqual(val, 1)

        # ((lambda f0, x0. f0(delta(a + x))) (lambda x1. x1, x))
        # = ((lambda x1. x1) delta(a + x)) = delta(a + x)
        diffeo = Shift((a,), (x,))
        apply_func = Function((f0, x0), App(f0, (Delta(diffeo),)))
        identity = App(apply_func, (Function((x1,), x1), x))
        expr = Int(identity, mu)
        var_val = VarVal({a.name: 3})
        gen = self.gen(expr, var_val)

        traces = extract_traces_from_name(gen, x.name)
        expected_trace = {(TraceName.Integral, TraceName.AppFun, TraceName.AppArg, TraceName.AppFun, TraceName.Leaf)}
        self.assertEqual(traces, expected_trace)  # TODO: BUG
        val = evaluate(expr, var_val, 1)
        self.assertEqual(val, 0)
        # (lambda x0, x1. delta(a + x0) + x) (x)
        # (lambda x0, x1. delta(a + x0) + delta(a + x1)) (x, x)

    def test_delta_in_function_app(self):
        a0 = Var("a0")
        # int (lambda a0. a0 * delta(y))(a) dy = int a * delta(y) dy
        a = Var("a")
        y = TegVar("y")
        diffeo = Shift((a0,), (y,))
        func = Function((a0,), a0 * Delta(diffeo))
        var_val = VarVal({a.name: 3})

        mu = BoundedLebesgue(-1, 1, y)
        expr = Int(App(func, (a,)), mu)
        gen = self.gen(expr, var_val)
        traces = extract_traces_from_name(gen, y.name)
        expected_trace = {
            (TraceName.Integral, TraceName.AppFun, TraceName.BinopRight, TraceName.AppFun, TraceName.Leaf)
        }
        self.assertEqual(traces, expected_trace)
        self.assertEqual(evaluate(expr, var_val, 1), 0)

        var_val = VarVal({a.name: 0.5})
        self.assertEqual(evaluate(expr, var_val, 1), 0.5)

        # ((lambda x0. delta(a - x0))(x) = delta(a - x)
        x0 = TegVar("x0")
        x = TegVar("x")
        diffeo = Shift((a,), (x0,))
        fun = Function((x0,), Delta(diffeo))
        mu = BoundedLebesgue(-1, 1, x)
        expr = Int(App(fun, (x,)), mu)
        # expr = Int(Delta(Shift((a,), (x,))), mu)
        var_val = VarVal({a.name: 3})
        gen = self.gen(expr, var_val)
        traces = extract_traces_from_name(gen, x.name)
        expected_traces = {(TraceName.Integral, TraceName.AppFun, TraceName.AppFun, TraceName.Leaf)}
        self.assertEqual(traces, expected_traces)
        self.assertEqual(evaluate(expr, var_val, 1), 0)

        var_val = VarVal({a.name: 0})
        self.assertEqual(evaluate(expr, var_val, 1), 1)

        # ((lambda f0, x0. f0(delta(a - x0))) (lambda x1. x1, x)) = delta(a - x)
        f0 = Var("f")
        x1 = Var("x1")
        fun = Function(
            (
                f0,
                x0,
            ),
            App(f0, (Delta(Shift((a,), (x0,))),)),
        )
        identity = Function((x1,), x1)
        expr = Int(App(fun, (identity, x)), mu)
        gen = self.gen(expr, var_val)
        traces = extract_traces_from_name(gen, x.name)
        expected_traces = {(TraceName.Integral, TraceName.AppFun, TraceName.AppArg, TraceName.AppFun, TraceName.Leaf)}
        self.assertEqual(traces, expected_traces)
        var_val = VarVal({a.name: 0})
        self.assertEqual(evaluate(expr, var_val, 1), 1)


class ScaleShift1(Affine):
    def function(self, vars: tuple[GExpr, ...], tvars: tuple[GExpr, ...]) -> tuple[GExpr, ...]:
        return tuple(tvar * var - Const(1) for var, tvar in zip(vars, tvars))

    def inverse(self, vars: tuple[GExpr, ...], tvars: tuple[GExpr, ...]) -> tuple[GExpr, ...]:
        return tuple((tvar + Const(1)) / var for var, tvar in zip(vars, tvars))


class TestTrickyDiffeos(TestCase):

    def test_cross_term_diffeo(self):
        x = TegVar("x")
        mu = BoundedLebesgue(-1, 1, x)
        t = Var("t")

        diffeo = ScaleByT((t,), (x,))

        # int (\y. y ? 1 : 0)(t * x - 1) dx
        y = TegVar("y")
        conditional = IfElse(y, Const(1), Const(0))
        func = Function((y,), conditional)
        integrand = App(func, (diffeo,))
        expr = Int(integrand, mu)

        dt = Var("dt")
        deriv_expr = deriv(expr, {t.name: dt.name})
        # [-1 \leq 1/t \leq 1] 1/t^2
        # t = 2 then expect 0.25

        var_val = VarVal({t.name: 2, dt.name: 1})
        # val = evaluate(expr, var_val, 1000)
        # print(val)
        val = evaluate(deriv_expr, var_val, 10)
        self.assertAlmostEqual(val, 0.25, 5)

    def test_2x_m2(self):
        x = TegVar("x")
        mu = BoundedLebesgue(-1, 1, x)
        t = Var("t")

        diffeo = Scale2ShiftT((t,), (x,))

        # int (\y. y ? 1 : 0)(2 * x - t) dx
        # 1 - t / 2
        y = TegVar("y")
        conditional = IfElse(y, Const(1), Const(0))
        func = Function((y,), conditional)
        integrand = App(func, (diffeo,))
        expr = Int(integrand, mu)

        dt = Var("dt")
        deriv_expr = deriv(expr, {t.name: dt.name})
        # - [-2 \leq t \leq 2] 1/2
        # t = 1 then expect -0.5

        var_val = VarVal({t.name: 1, dt.name: 1})
        # val = evaluate(expr, var_val, 5000)
        # self.assertAlmostEqual(val, 0.5, 1)
        deriv_val = evaluate(deriv_expr, var_val, 10)
        self.assertAlmostEqual(deriv_val, -0.5, 5)

    def test_quadratic(self):
        # t^2 x^2 - 1 = (tx - 1) (tx + 1)
        x = TegVar("x")
        mu = BoundedLebesgue(0, 5, x)
        t = Var("t")

        diffeo = QuadraticSquaredMinus1((t,), (x,))

        # int (\y. y ? 1 : 0)((tx - 1) (tx + 1)) dx
        # [ tx - 1 > 0] [ tx + 1 > 0]
        # [1 \leq t] (2 - 2 / t) + 2 [t \leq 0]
        y = TegVar("y")
        conditional = IfElse(y, Const(1), Const(0))
        func = Function((y,), conditional)
        integrand = App(func, (diffeo,))
        expr = Int(integrand, mu)

        dt = Var("dt")
        deriv_expr = deriv(expr, {t.name: dt.name})
        # [1 \leq t] 2/t^2
        # t = 2 then expect 0.5

        var_val = VarVal({t.name: 2, dt.name: 1})
        # val = evaluate(expr, var_val, 5000)
        # self.assertAlmostEqual(val, 0.5, 1)
        deriv_val = evaluate(deriv_expr, var_val, 10)
        self.assertAlmostEqual(deriv_val, 0.25, 5)

    @unittest.skip("Degeneracy")
    def test_square(self):
        # t^2 x^2 - 2tx + 1 = (tx - 1)^2
        x = TegVar("x")
        mu = BoundedLebesgue(-1, 1, x)
        t = Var("t")

        diffeo = SumOfSquares((t,), (x,))

        # int (\y. y ? 1 : 0)(2 * x - t) dx
        # 1 - t / 2
        y = TegVar("y")
        conditional = IfElse(y, Const(1), Const(0))
        func = Function((y,), conditional)
        integrand = App(func, (diffeo,))
        expr = Int(integrand, mu)

        dt = Var("dt")
        deriv_expr = deriv(expr, {t.name: dt.name})
        # - [-2 \leq t \leq 2] 1/2
        # t = 1 then expect -0.5

        var_val = VarVal({t.name: 1.5, dt.name: 1})
        # val = evaluate(expr, var_val, 5000)
        # self.assertAlmostEqual(val, 0.5, 1)
        deriv_val = evaluate(deriv_expr, var_val, 10)
        self.assertAlmostEqual(deriv_val, -0.5, 5)

    def test_multidimensional(self):
        # int int H((y - 2x, y)) dx dy
        # = int int delta((y - 2x + t, y)) dx dy
        # w = y - 2x + t, z = y
        # int int delta((w, z)) dw dz / 2
        # int_{-1}^{1} dz / 2
        x = TegVar("x")
        y = TegVar("y")
        mu = BoundedLebesgue(-1, 1, x)
        nu = BoundedLebesgue(-1, 1, y)
        t = Var("t")

        diffeo = SquareSlice((t,), (x, y))

        # int (\w, z. w ? 1 : 0)((y - 2x, y)) dx
        w = TegVar("w")
        z = TegVar("z")
        conditional = IfElse(w, Const(1), Const(0))
        func = Function((w, z), conditional)
        integrand = App(func, (diffeo,))
        expr = Int(Int(integrand, mu), nu)

        dt = Var("dt")
        deriv_expr = deriv(expr, {t.name: dt.name})

        var_val = VarVal({t.name: 1, dt.name: 1})
        # val = evaluate(expr, var_val, 5000)
        # self.assertAlmostEqual(val, 0.5, 1)
        deriv_val = evaluate(deriv_expr, var_val, 10)
        self.assertAlmostEqual(deriv_val, 1, 5)

    def test_multidimensional_circ(self):
        # QuarterCircle
        # z1 = t - x ^ 2 - y ^ 2
        x = TegVar("x")
        y = TegVar("y")
        mu = BoundedLebesgue(0, 1, x)
        nu = BoundedLebesgue(-1, 1, y)
        t = Var("t")

        diffeo = QuarterCircle((t,), (x, y))

        # int (\w, z. w ? 1 : 0)((y - 2x, y)) dx
        w = TegVar("w")
        z = TegVar("z")
        conditional = IfElse(w, Const(1), Const(0))
        func = Function((w, z), conditional)
        integrand = App(func, (diffeo,))
        expr = Int(Int(integrand, mu), nu)

        dt = Var("dt")
        deriv_expr = deriv(expr, {t.name: dt.name})

        var_val = VarVal({t.name: 1, dt.name: 1})
        # val = evaluate(expr, var_val, 5000)
        # self.assertAlmostEqual(val, 0.5, 1)
        deriv_val = evaluate(deriv_expr, var_val, 5000)
        self.assertAlmostEqual(deriv_val, np.pi / 2, 1)

    def test_nested_diffeos(self):
        # 2x - t \circ (tx - 1)
        x = TegVar("x")
        mu = BoundedLebesgue(0, 2, x)
        t = Var("t")
        z = TegVar("z")

        inner_diffeo = Scale2ShiftT((t,), (z,))
        diffeo = QuadraticSquaredMinus1((t,), (x,))

        # int (\z. (\y. y ? 1 : 0)(2 * z - t))(t^2x^2 - 1) dx
        y = TegVar("y")
        conditional = IfElse(y, Const(1), Const(0))
        func = Function((y,), conditional)
        inner = App(func, (inner_diffeo,))
        outer_func = Function((z,), inner)
        integrand = App(outer_func, (diffeo,))
        expr = Int(integrand, mu)

        dt = Var("dt")
        deriv_expr = deriv(expr, {t.name: dt.name})
        # - [-2 \leq t \leq 2] 1/2
        # t = 1 then expect -0.5

        var_val = VarVal({t.name: 1, dt.name: 1})
        # val = evaluate(expr, var_val, 5000)
        # self.assertAlmostEqual(val, 0.5, 1)
        deriv_val = evaluate(deriv_expr, var_val, 1)
        self.assertAlmostEqual(deriv_val, 5 / (2 * np.sqrt(6)), 2)

    def test_nested_diffeos_cross(self):
        # (tx)^2 - 1 \circ (tx)^2 - 1
        x = TegVar("x")
        mu = BoundedLebesgue(0, 3, x)
        t = Var("t")
        z = TegVar("z")

        inner_diffeo = QuadraticSquaredMinus1((t,), (z,))
        diffeo = QuadraticSquaredMinus1((t,), (x,))

        # int (\z. (\y. y ? 1 : 0)(tz - 1))(tx - 1) dx
        y = TegVar("y")
        conditional = IfElse(y, Const(1), Const(0))
        func = Function((y,), conditional)
        inner = App(func, (inner_diffeo,))
        outer_func = Function((z,), inner)
        integrand = App(outer_func, (diffeo,))
        expr = Int(integrand, mu)

        dt = Var("dt")
        deriv_expr = deriv(expr, {t.name: dt.name})
        # - [-2 \leq t \leq 2] 1/2
        # t = 1 then expect -0.5

        var_val = VarVal({t.name: 1, dt.name: 1})
        # val = evaluate(expr, var_val, 5000)
        # self.assertAlmostEqual(val, 0.5, 1)
        deriv_val = evaluate(deriv_expr, var_val, 100)
        self.assertAlmostEqual(deriv_val, 5 / (2 * np.sqrt(2)), 2)

    def test_nested_2d_affine(self):
        # (z - 2w + t, y) \circ (y - 2x + t, y)
        x, y = TegVar("x"), TegVar("y")
        mu = BoundedLebesgue(0, 2, x)
        nu = BoundedLebesgue(0, 1, y)
        t = Var("t")
        w, z = TegVar("w"), TegVar("z")

        inner_diffeo = SquareSlice((t,), (w, z))
        diffeo = SquareSlice((t,), (x, y))

        # int int (\w, z. (\p, q. p ? 1 : 0)((z - 2w + t, z)))((y - 2x + t, y)) dxdy
        p, q = TegVar("p"), TegVar("q")
        conditional = IfElse(p, Const(1), Const(0))
        func = Function((p, q), conditional)
        inner = App(func, (inner_diffeo,))
        outer_func = Function((w, z), inner)
        integrand = App(outer_func, (diffeo,))
        expr = Int(Int(integrand, mu), nu)

        dt = Var("dt")
        deriv_expr = deriv(expr, {t.name: dt.name})
        # - [-2 \leq t \leq 2] 1/2
        # t = 1 then expect -0.5

        var_val = VarVal({t.name: 1, dt.name: 1})
        # val = evaluate(expr, var_val, 5000)
        # self.assertAlmostEqual(val, np.pi / 4, 1)
        deriv_val = evaluate(deriv_expr, var_val, 100)
        self.assertAlmostEqual(deriv_val, -0.25, 2)

    def test_nested_2d(self):
        # (t - w^2 - z^2, y) \circ (x + t, y)
        x, y = TegVar("x"), TegVar("y")
        mu = BoundedLebesgue(-1, 2, x)
        nu = BoundedLebesgue(0, 1, y)
        t = Var("t")
        w, z = TegVar("w"), TegVar("z")

        inner_diffeo = QuarterCircle((t,), (w, z))
        diffeo = ShiftByT2D((t,), (x, y))

        # int int (\w, z. (\p, q. p ? 1 : 0)((t - w^2 - z^2, z)))((x + t, y)) dxdy
        p, q = TegVar("p"), TegVar("q")
        conditional = IfElse(p, Const(1), Const(0))
        func = Function((p, q), conditional)
        inner = App(func, (inner_diffeo,))
        outer_func = Function((w, z), inner)
        integrand = App(outer_func, (diffeo,))
        expr = Int(Int(integrand, mu), nu)

        dt = Var("dt")
        deriv_expr = deriv(expr, {t.name: dt.name})
        # - [-2 \leq t \leq 2] 1/2
        # t = 1 then expect -0.5

        var_val = VarVal({t.name: 1, dt.name: 1})
        # val = evaluate(expr, var_val, 5000)
        # self.assertAlmostEqual(val, np.pi / 4, 1)
        deriv_val = evaluate(deriv_expr, var_val, 3000)
        self.assertAlmostEqual(deriv_val, np.pi / 4 - 1, 1)

    # def test_nested_2d(self):
    #     # (t - w^2 - z^2, y) \circ (y - 2x + t, y)
    #     x, y = TegVar("x"), TegVar("y")
    #     mu = BoundedLebesgue(0, 2, x)
    #     nu = BoundedLebesgue(0, 1, y)
    #     t = Var("t")
    #     w, z = TegVar("w"), TegVar("z")
    #
    #     inner_diffeo = QuarterCircle((t,), (w, z))
    #     diffeo = SquareSlice((t,), (x, y))
    #
    #     # int int (\w, z. (\p, q. p ? 1 : 0)((t - w^2 - z^2, z)))((x + t, y)) dxdy
    #     p, q = TegVar("p"), TegVar("q")
    #     conditional = IfElse(p, Const(1), Const(0))
    #     func = Function((p, q), conditional)
    #     inner = App(func, (inner_diffeo,))
    #     outer_func = Function((w, z), inner)
    #     integrand = App(outer_func, (diffeo,))
    #     expr = Int(Int(integrand, mu), nu)
    #
    #     dt = Var("dt")
    #     deriv_expr = deriv(expr, {t.name: dt.name})
    #     # - [-2 \leq t \leq 2] 1/2
    #     # t = 1 then expect -0.5
    #
    #     var_val = VarVal({t.name: 1, dt.name: 1})
    #     # val = evaluate(expr, var_val, 5000)
    #     # self.assertAlmostEqual(val, np.pi / 4, 1)
    #     deriv_val = evaluate(deriv_expr, var_val, 5000)
    #     self.assertAlmostEqual(deriv_val, np.pi / 4, 2)

    @unittest.skip("domain error")
    def test_nested_2d_swapped(self):
        # (w - 2z + t, y) \circ  (t - x^2 - y^2, y)
        x, y = TegVar("x"), TegVar("y")
        mu = BoundedLebesgue(0, 1, x)
        nu = BoundedLebesgue(-1, 0, y)
        t = Var("t")
        w, z = TegVar("w"), TegVar("z")

        inner_diffeo = SquareSlice2((t,), (w, z))
        diffeo = QuarterCircle((t,), (x, y))

        # int int (\w, z. (\p, q. p ? 1 : 0)((t - w^2 - z^2, z)))((y - 2x + t, y)) dxdy
        p, q = TegVar("p"), TegVar("q")
        conditional = IfElse(p, Const(1), Const(0))
        func = Function((p, q), conditional)
        inner = App(func, (inner_diffeo,))
        outer_func = Function((w, z), inner)
        integrand = App(outer_func, (diffeo,))
        expr = Int(Int(integrand, mu), nu)

        dt = Var("dt")
        deriv_expr = deriv(expr, {t.name: dt.name})
        # - [-2 \leq t \leq 2] 1/2
        # t = 1 then expect -0.5

        var_val = VarVal({t.name: 0.1, dt.name: 1})
        # val = evaluate(expr, var_val, 5000)
        # self.assertAlmostEqual(val, np.pi / 4, 1)
        deriv_val = evaluate(deriv_expr, var_val, 5000)
        self.assertAlmostEqual(deriv_val, np.pi / 4, 2)


if __name__ == "__main__":
    unittest.main()
