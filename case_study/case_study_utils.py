import functools
from collections import defaultdict
from dataclasses import dataclass
from numbers import Number
from unittest import TestCase
import numpy as np
import sys
from typing import Callable
from inspect import signature
import scipy.optimize as spop
import matplotlib.pyplot as plt

from potto import Shift, Affine
from potto import (
    Delta,
    Measure,
    Var,
    TegVar,
    Heaviside,
    GExpr,
    Const,
    Int,
    Sym,
    Function,
    App,
)
from potto import Sqrt, Sqr
from potto import deriv
from potto import evaluate, evaluate_all
from potto import VarVal
from potto import simplify
from time import time

# trial 1
# empirical_risk_minimization deriv: 0.0004360675811767578
# normal_pdf_sigma5 deriv: 0.00012087821960449219
# trunc_normal deriv: 0.0003268718719482422
# trial 2
# empirical_risk_minimization deriv: 0.00032806396484375
# normal_pdf_sigma5 deriv: 9.202957153320312e-05
# trunc_normal deriv: 0.00023293495178222656
# trial 3
# empirical_risk_minimization deriv: 0.0003657341003417969
# normal_pdf_sigma5 deriv: 0.000102996826171875
# trunc_normal deriv: 0.00023293495178222656
# on average: 746 microseconds
# vs. double counting empirical 1122 microseconds
# potto_comp_times = [0.0004360675811767578 + 0.00012087821960449219 + 0.0003268718719482422, 0.00032806396484375 + 9.202957153320312e-05 + 0.00023293495178222656, 0.00023293495178222656 + 0.000102996826171875 + 0.00023293495178222656]

# time to evaluate the derivative
# each partial
# runtime: 0.42827701568603516
# runtime: 0.44319891929626465
# runtime: 0.527947187423706
# total: 1.3994231224060059
# each partial
# runtime: 0.4598698616027832
# runtime: 0.4458780288696289
# runtime: 0.4395768642425537
# total: 1.3453247547149658
# each partial
# runtime: 0.43221187591552734
# runtime: 0.42958593368530273
# runtime: 0.4425499439239502
# total: 1.3043477535247803
# on average: 1.349698543548584
# potto = [1.3994231224060059, 1.3453247547149658, 1.349698543548584]
# ~ 1.35 seconds

empirical_risk_minimization = np.mean([0.0004360675811767578, 0.00032806396484375, 0.0003657341003417969])
normal_pdf_sigma5 = np.mean([0.00012087821960449219, 9.202957153320312e-05, 0.000102996826171875])
trunc_normal = np.mean([0.0003268718719482422, 0.00023293495178222656, 0.00023293495178222656])


def potto():
    def potto_decorator(f: Callable[[Var | TegVar, ...], GExpr]):
        @functools.wraps(f)
        def wrapped_f():
            params = signature(f).parameters
            args = [
                TegVar(f"{a}{i}") if v.annotation == TegVar else Var(f"{a}{i}")
                for i, (a, v) in enumerate(params.items())
            ]
            darg_names = [Sym(f"d{a}{i}") for i, a in enumerate(params)]
            e = Function(tuple(args), f(*args))
            e = simplify(e)

            ctx = {a.name: da_name for a, da_name in zip(args, darg_names)}
            start = time()
            # when delta=True, account for Dirac deltas
            # when delta=False, ignore Dirac deltas
            de = deriv(e, ctx, delta=False)
            end = time()
            print(f"{f.__name__} deriv: {end - start}")
            de = simplify(de)
            return e, de

        return wrapped_f

    return potto_decorator


def list_to_param(params, vals):
    return {k.name: v for k, v in zip(params, vals)}


def eval_expr(e, ps, params):
    param_vals = list_to_param(params, ps)
    ret = evaluate_all(e, VarVal(param_vals), num_samples=50)
    # print(f'eval: {ps}\n\t: {ret}')
    return ret


def eval_grad_expr(de, ps, params, dparams):
    grad = []
    for dparam in dparams[:1]:
        differential = {dp.name: 0 for dp in dparams}
        differential[dparam.name] = 1
        dparam_vals = list_to_param(params, ps) | differential
        start = time()
        dc_eval = evaluate_all(de, VarVal(dparam_vals), num_samples=50)
        end = time()
        print("runtime", end - start)

        grad.append(dc_eval)
    # print(f'grad: {ps}\n\t: {grad}')
    return grad[0]
