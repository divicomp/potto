from potto import Const, Var, TegVar, IfElse, FlipShift, Int, BoundedLebesgue
from potto import deriv, evaluate, VarVal

# Declare the variable of integration, variable, and infinitesimal
x, t, dt = TegVar("x"), Var("t"), Var("dt")

# integral from 0 to 1 of (if x - t >= 0 then 1 else 0) dx
integrand = IfElse(FlipShift((t,), (x,)), Const(1), Const(0))
mu = BoundedLebesgue(0, 1, x)
expr = deriv(Int(integrand, mu), {t.name: dt.name})

# Evaluate the derivative of the integral at t = 0.5, dt = 1
ctx = VarVal({t.name: 0.5, dt.name: 1})
print("Dₜ∫₀¹ [x ≤ t] dx at t=0.5 is", evaluate(expr, ctx))
