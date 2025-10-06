import unittest
import math
import numpy as np

from potto.lang.grammar import (
    TegVar, 
    Const, 
    SingularDivision, 
    Int,
    Var,
    SingularDivision
)
from potto.lang.evaluate import evaluate
from potto.lang.derivative import deriv
from potto.lang.evaluate_utils import VarVal
from potto import BoundedLebesgue
from potto.libs.pmath import Exp, Sin, Cos



class TestSingularIntegration(unittest.TestCase):
    """All tests below integrate using Int over intervals that include the singularity.
    
    These tests are adapted from the distro project's test_evaluate.py TestSingularIntegration class.
    They test Cauchy principal value integrals and Hadamard finite part integrals using 
    Potto's SingularDivision construct.
    """

    def test_constant_over_linear_k1(self) -> None:
        """C∫_0^2 3/(x-1) dx = 3 (C∫_0^2 1/(x-1) dx) = 0"""
        x = TegVar("x")
        s = Var("s")
        integrand = SingularDivision(numerator=Const(3.0), x=x, s=s, power=1)
        measure = BoundedLebesgue(Const(0.0), Const(2.0), x)
        integ = Int(integrand=integrand, measure=measure)
        estimate = evaluate(integ, num_samples=1000, env_or_var_val=VarVal({s.name: 1.0}))
        self.assertAlmostEqual(estimate, 0, delta=1e-2)

    def test_pow1_x_over_x_minus_half(self) -> None:
        """C ∫_0^1 x/(x-0.5) dx = 1.0"""
        x = TegVar("x")
        s = Var("s")
        integrand = SingularDivision(numerator=x, x=x, s=s, power=1)
        measure = BoundedLebesgue(Const(0.0), Const(1.0), x)
        integ = Int(integrand=integrand, measure=measure)
        estimate = evaluate(integ, num_samples=5000, env_or_var_val=VarVal({s.name: 0.5}))
        self.assertAlmostEqual(estimate, 1.0, delta=1e-2)

    def test_pow2_x_over_x_minus_half_squared(self) -> None:
        """H ∫_0^1 x/(x-0.5)^2 dx = -2.0"""
        x = TegVar("x")
        s = Var("s")
        integrand = SingularDivision(numerator=x, x=x, s=s, power=2)
        measure = BoundedLebesgue(Const(0.0), Const(1.0), x)
        integ = Int(integrand=integrand, measure=measure)
        estimate = evaluate(integ, num_samples=5000, env_or_var_val=VarVal({s.name: 0.5}))
        self.assertAlmostEqual(estimate, -2.0, delta=1e-2)

    def test_pow1_one_over_x_minus_point_one(self) -> None:
        """C ∫_0^1 1/(x-0.1) dx = log((1-0.1)/0.1) = log(9)"""
        x = TegVar("x")
        s = Var("s")
        integrand = SingularDivision(numerator=Const(1.0), x=x, s=s, power=1)
        measure = BoundedLebesgue(Const(0.0), Const(1.0), x)
        integ = Int(integrand=integrand, measure=measure)
        estimate = evaluate(integ, num_samples=10000, env_or_var_val=VarVal({s.name: 0.1}))
        expected = math.log(9.0)
        self.assertAlmostEqual(estimate, expected, delta=1e-1)

    def test_pow1_x_over_x_minus_point_one(self) -> None:
        """C ∫_0^1 x/(x-0.1) dx = 1 + 0.1 * log(9)"""
        x = TegVar("x")
        s = Var("s")
        integrand = SingularDivision(numerator=x, x=x, s=s, power=1)
        measure = BoundedLebesgue(Const(0.0), Const(1.0), x)
        integ = Int(integrand=integrand, measure=measure)
        estimate = evaluate(integ, num_samples=5000, env_or_var_val=VarVal({s.name: 0.1}))
        expected = 1.0 + 0.1 * math.log(9.0)
        self.assertAlmostEqual(estimate, expected, delta=1e-2)


class TestSingularIntegrationDerivPotto(unittest.TestCase):
    """Potto-style versions of derivative/Hilbert transform tests.

    These tests build Potto expressions using `TegVar`, `Const`, `SingularDivision`, and `Int`.
    We avoid relying on non-Potto syntax like Variable/Operation/VectorIntegral.
    """

    def test_derivative_theta_pow2_x_over_x_minus_half_squared(self) -> None:
        """H ∫_0^1 x/(x - 0.5)^2 dx = -2.0"""
        x = TegVar("x")
        s = Var("s")
        integrand = SingularDivision(numerator=x, x=x, s=s, power=2)
        measure = BoundedLebesgue(Const(0.0), Const(1.0), x)
        integ = Int(integrand=integrand, measure=measure)
        estimate = evaluate(integ, num_samples=2000, env_or_var_val=VarVal({s.name: 0.5}))
        self.assertAlmostEqual(estimate, -2.0, delta=1e-2)

    def test_derivative_theta_vector_pow1_linear_over_x_minus_point_one(self) -> None:
        """
        For numerator (t0 x + t1) and k=1, z=0.1:
        PV ∫ x/(x - 0.1) = 1 + 0.1 log 9
        PV ∫ 1/(x - 0.1) = log 9
        """
        x = TegVar("x")
        s = Var("s")
        measure = BoundedLebesgue(Const(0.0), Const(1.0), x)
        integ_x = Int(SingularDivision(x, x, s, 1), measure)
        val_x = evaluate(integ_x, num_samples=10000, env_or_var_val=VarVal({s.name: 0.1}))
        expected_x = 1.0 + 0.1 * math.log(9.0)
        self.assertAlmostEqual(val_x, expected_x, delta=1e-2)

        integ_one = Int(SingularDivision(Const(1.0), x, s, 1), measure)
        val_one = evaluate(integ_one, num_samples=1000, env_or_var_val=VarVal({s.name: 0.1}))
        expected_one = math.log(9.0)
        self.assertAlmostEqual(val_one, expected_one, delta=1e-1)

    def test_hilbert_transform(self) -> None:
        """
        Compare -(1/pi) * PV ∫ f(x)/(x-s) on (-1, 1) at s=0.5 against known values (subset).
        Functions tested: x, x^2, exp(x).
        """

        x = TegVar("x")
        s = Var("s")
        bounds = (-1.0, 1.0)
        funcs = [x, x * x, Exp(x), x * Exp(x), Sin(x), Cos(x)]
        ground_truths = [-0.462, -0.231, -0.291, -0.894, -0.409, 0.458]

        for f, truth in zip(funcs, ground_truths):
            measure = BoundedLebesgue(Const(bounds[0]), Const(bounds[1]), x)
            integ = -Const(1.0 / math.pi) * Int(SingularDivision(f, x, s, 1), measure)
            pv_value = evaluate(integ, num_samples=1000, env_or_var_val=VarVal({s.name: 0.5}))
            h_value = pv_value
            self.assertAlmostEqual(h_value, truth, delta=1e-2)

    def test_hilbert_transform_derivative_wrt_singularity(self) -> None:
        """
        Differentiate w.r.t. singularity s via finite differences:
        d/ds [-(1/pi) PV ∫ f(x)/(x-s) dx] ≈ (H(s+eps) - H(s-eps)) / (2 eps)
        Functions tested: x, x^2, exp(x).
        """
        x = TegVar("x")
        s = Var("s")
        ds = Var("ds")
        bounds = (-1.0, 1.0)
        funcs = [x, x * x, Exp(x), x * Exp(x), Sin(x), Cos(x)]
        deriv_ground_truths = [0.774, -0.074714029, 1.51771, 0.467989, 0.815591, 0.8675]

        def hilbert_at(s: Var, f) -> float:
            measure = BoundedLebesgue(Const(bounds[0]), Const(bounds[1]), x)
            return -Const(1.0 / math.pi) * Int(SingularDivision(f, x, s, 1), measure)

        for f, truth in zip(funcs, deriv_ground_truths):
            d_h_expr = deriv(hilbert_at(s, f), {s.name: ds.name})
            d_h = evaluate(d_h_expr, num_samples=10000, env_or_var_val=VarVal({s.name: 0.5, ds.name: 1.0}))
            self.assertAlmostEqual(d_h, truth, delta=1e-2)


class TestSingularIntegrationProduct(unittest.TestCase):
    def test_product(self):
        x = TegVar("x")
        s = Var("s")
        integrand = SingularDivision(numerator=x * x, x=x, s=s, power=2)
        measure = BoundedLebesgue(Const(0.0), Const(1.0), x)
        integ = Int(integrand=integrand, measure=measure)
        estimate = evaluate(integ, num_samples=50, env_or_var_val=VarVal({s.name: 0.5}))
        # H ∫_0^1 (x^2 / (x - s)^2) dx
        # = C ∫_0^1 (2x / (x - s)) dx − [ x^2/(x - s) ]_0^1
        # = C ∫_0^1 (2 + 2s/(x - s)) dx − (1/(1 - s) − 0)
        # = 2 + 2s · C ∫_0^1 (1/(x - s)) dx − 1/(1 - s)
        # = 2 + 2s · log((1 - s)/s) − 1/(1 - s)  for s ∈ (0,1)
        # In particular, at s = 1/2 the value is 0.
        self.assertAlmostEqual(estimate, 0.0, delta=1e-2)

        # # (x^2 / (x - s)^2)
        # integ = Int(x * SingularDivision(numerator=x, x=x, s=s, power=2), measure)
        # self.assertAlmostEqual(0, estimate1, delta=1e-2)


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
