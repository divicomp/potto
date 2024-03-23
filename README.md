# Potto
Potto is a differentiable programming language that has an integral primitive and supports higher-order functions. This repository contains the core library implementation, while the applications can be found at https://github.com/divicomp/potto_applications. The applications include differentiating a ray tracing renderer.

## Installation Instructions
Potto requires Python 3.10+. To install Potto, run:
```
gh repo clone martinjm97/potto
cd potto
pip install -e .
```

## Illustrative Example
We present the example depicted in Figure 1 of the paper:

![D_t \int_{x = 0}^1 [x \leq t]](https://latex.codecogs.com/svg.latex?D_t%20\int_{x%20=%200}^1%20[x%20\leq%20t]),

which is the the derivative of the integral of a jump discontinuity that is `1` if `x <= t` and `0` otherwise.

At `t=0.5`, the the result is `1`. However, discretizing before computing the derivative as is standard in differentiable programming languages (e.g., PyTorch and TensorFlow) results in a derivative of 0. 

In Potto, we can implement this example with:
```python
from potto import (
    Const,
    Var,
    TegVar,
    IfElse,
    FlipShift,
    Int,
    deriv,
    evaluate_all,
    BoundedLebesgue,
    VarVal,
)

# Declare the variable of integration, variable, and infinitesimal
x, t, dt = TegVar("x"), Var("t"), Var("dt")

# integral from 0 to 1 of (if x - t >= 0 then 1 else 0) dx
integrand = IfElse(FlipShift((t,), (x,)), Const(1), Const(0))
mu = BoundedLebesgue(0, 1, x)
expr = deriv(Int(integrand, mu), {t.name: dt.name})

# Evaluate the derivative of the integral at t = 0.5, dt = 1
ctx = VarVal({t.name: 0.5, dt.name: 1})
print("Dₜ∫₀¹ [x ≤ t] dx at t=0.5 is", evaluate_all(expr, ctx))
```


## Code structure
The `potto` folder contains the core implementation. We break the implementation into pieces:
- `ir` contains the code for an IR lowering that we do to abstract diffeomorphisms outside of Dirac deltas and conditionals, as well as add more information to diffeomorphisms to make evaluation easier.
- `lang` contains the core language implementation such as the grammar, evaluator, and derivative code.
- `libs` contains useful libraries for diffeomorphisms, measures, and math operators.

## Figures

Figure 2 is at `case_study/plot_normal.py`.
Figure 3 is at `case_study/figure3.py`.
## TODO: Citation
