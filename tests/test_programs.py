import math
import unittest

from potto.lang.grammar import Var, TegVar, Const, Assign, Seq, IfPos, IfElse, Int, SingularDivision
from potto.lang.evaluate import evaluate_program, evaluate
from potto.lang.evaluate_utils import Environment
from potto.libs.measure import Uniform, BoundedLebesgue
from potto.libs.pmath import Sin, Exp
from potto.lang.derivative import deriv
from potto.lang.grammar import Sym


class TestPottoPrograms(unittest.TestCase):
    def test_assign(self) -> None:
        # Program: x = x + 2
        x = Var("x")
        env = Environment({x.name: 1.0})
        program = Assign(target=x, expr=x + Const(2.0))
        gamma = evaluate_program(program, env)
        self.assertEqual(gamma[x.name], 3.0)

        dx = Var("dx")
        dprogram = deriv(program, {x.name: dx.name})
        dgamma = evaluate_program(dprogram, Environment({x.name: 1.0, dx.name: 1.0}))
        # Derivatives: dx2/dx1 = dx1 = 1 
        self.assertAlmostEqual(dgamma[dx.name], 1.0, delta=1e-12)

    def test_seq_primal(self) -> None:
        # Program: x = x + 2; y = x * 5
        x = Var("x")
        y = Var("y")
        env = Environment({x.name: 1.0})
        program = Seq(Assign(target=x, expr=x + Const(2.0)), Assign(target=y, expr=x * Const(5.0)))
        gamma = evaluate_program(program, env)
        self.assertEqual(gamma[x.name], 3.0)
        self.assertEqual(gamma[y.name], 15.0)

        dx = Var("dx")
        dy = Var("dy")
        dprogram = deriv(program, {x.name: dx.name})
        dgamma = evaluate_program(dprogram, Environment({x.name: 1.0, dx.name: 1.0}))

    def test_ifpos_primal(self) -> None:
        # Program: ifpos(sin(x)) { y = 42 } else { y = -1 }
        x = Var("x")
        y = Var("y")
        env = Environment({x.name: 1.0})
        program = IfPos(condition=Sin(x), then_branch=Assign(y, Const(42.0)), else_branch=Assign(y, Const(-1.0)))
        gamma = evaluate_program(program, env)
        self.assertEqual(gamma[y.name], 42.0)

        env = Environment({x.name: -1.0})
        program = IfPos(condition=Sin(x), then_branch=Assign(y, Const(42.0)), else_branch=Assign(y, Const(-1.0)))
        gamma = evaluate_program(program, env)
        self.assertEqual(gamma[y.name], -1.0)

    def test_program_with_integral_assignment_simple(self) -> None:
        # Integral: ∫_{0}^1 1/(x-0.1) dx = PV log(9).
        x = TegVar("x")
        y = Var("y")
        s = Var("s")

        integ = Int(
            integrand=SingularDivision(numerator=Const(1.0), x=x, s=s, power=1),
            measure=BoundedLebesgue(0.0, 1.0, x),
        )

        prog = Seq(
            Assign(target=s, expr=Const(0.1)),
            Assign(target=y, expr=integ),
        )

        gamma = evaluate_program(prog, num_samples=4000)
        self.assertEqual(gamma[s.name], 0.1)
        self.assertAlmostEqual(gamma[y.name], math.log(9.0), delta=8e-2)

    def test_program_with_nested_integrals_and_conditionals_runs(self) -> None:
        # Exercise nested integrals and conditionals
        x = TegVar("x")
        w = TegVar("w")
        y = Var("y")
        z = Var("z")
        t = Var("t")

        # Inner 1D integral over x ∈ [-1,1]: x + 1/x + ifpos(x + t) { e^{x t} } else { t*x }
        # Use SingularDivision for the 1/x term since it has a singularity at x=0
        s_zero = Var("s_zero")
        inner_integrand = (x + SingularDivision(numerator=Const(1.0), x=x, s=s_zero, power=1)) + IfElse(x + t, Exp(x * t), t * x)
        inner = Int(integrand=inner_integrand, measure=BoundedLebesgue(-1.0, 1.0, x))

        # y = t + inner
        y_assign = Assign(target=y, expr=t + inner)

        # z = ∫_{w∈[2,3]} [ ifpos(x + y + t) { e^{x t} } else { t*x } ] dw
        # Note: x is integrated separately inside this integrand
        inner_x = Int(integrand=IfElse(x + y + t, Exp(x * t), t * x), measure=BoundedLebesgue(0.0, 1.0, x))
        two_d = Int(integrand=inner_x, measure=BoundedLebesgue(2.0, 3.0, w))
        z_assign = Assign(target=z, expr=two_d)

        program = Seq(y_assign, z_assign)

        env = Environment({t.name: 0.5, z.name: 1.0, s_zero.name: 0.0})
        gamma = evaluate_program(program, env, num_samples=10000)
        # y = t + ∫_{-1}^1 [x + PV(1/x) + ifpos(x + t){e^{xt}} else {t·x}] dx
        #   = 0.5 + 0 + 0 + (∫_{-0.5}^1 e^{0.5x} dx + ∫_{-1}^{-0.5} 0.5x dx) ≈ 2.052
        self.assertAlmostEqual(gamma[y.name], 2.052, delta=5e-2)
        # z = ∫_{w=2}^3 [ ∫_{x=0}^1 ifpos(x + y + t){e^{xt}} else {t·x} dx ] dw
        # Since y + t > 0, inner = ∫_0^1 e^{0.5x} dx = 2(e^{0.5} − 1) ≈ 1.297; outer interval has length 1
        self.assertAlmostEqual(gamma[z.name], 1.297, delta=6e-2)

    def test_program_forward_derivative_seq_chain_primal_values(self) -> None:
        # Program: x = p^2; y = x + p
        p = Var("p")
        x = Var("x")
        y = Var("y")
        env = Environment({p.name: 2.0})
        program = Seq(
            Assign(target=x, expr=(p ** 2)),
            Assign(target=y, expr=x + p),
        )
        gamma = evaluate_program(program, env)
        self.assertAlmostEqual(gamma[x.name], 4.0, delta=1e-12)
        self.assertAlmostEqual(gamma[y.name], 6.0, delta=1e-12)
        # Derivative checks from reference
        dp = Sym("dp")
        dx_expr = deriv((p ** 2), {p.name: dp})
        dy_expr = deriv(((p ** 2) + p), {p.name: dp})
        denv = Environment({p.name: 2.0, dp: 1.0})
        self.assertAlmostEqual(evaluate(dx_expr, denv), 4.0, delta=1e-12)
        self.assertAlmostEqual(evaluate(dy_expr, denv), 5.0, delta=1e-12)

    def test_program_forward_derivative_ifpos_then_primal_values(self) -> None:
        # Program: ifpos(sin(x)) { z = 2 * p } else { z = 0 }
        x = Var("x")
        p = Var("p")
        z = Var("z")
        env = Environment({x.name: 1.0, p.name: 3.0})  # sin(1) > 0 → then_branch
        program = IfPos(
            condition=Sin(x),
            then_branch=Assign(z, Const(2.0) * p),
            else_branch=Assign(z, Const(0.0)),
        )
        gamma = evaluate_program(program, env)
        self.assertAlmostEqual(gamma[z.name], 6.0, delta=1e-12)
        # Derivative check from reference
        dp = Sym("dp")
        dz_expr = deriv(Const(2.0) * p, {p.name: dp})
        denv = Environment({p.name: 3.0, dp: 1.0})
        self.assertAlmostEqual(evaluate(dz_expr, denv), 2.0, delta=1e-12)

    def test_program_forward_derivative_explicit_param_primal_values(self) -> None:
        # Program: w = p + 3
        p = Var("p")
        w = Var("w")
        env = Environment({p.name: 5.0})
        program = Assign(target=w, expr=p + Const(3.0))
        gamma = evaluate_program(program, env)
        self.assertAlmostEqual(gamma[w.name], 8.0, delta=1e-12)
        # Derivative check from reference
        dp = Sym("dp")
        dw_expr = deriv(p + Const(3.0), {p.name: dp})
        denv = Environment({p.name: 5.0, dp: 1.0})
        self.assertAlmostEqual(evaluate(dw_expr, denv), 1.0, delta=1e-12)


if __name__ == "__main__":
    unittest.main()


