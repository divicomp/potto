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


def potto():
    def potto_decorator(f: Callable[[Var | TegVar, ...], GExpr]):
        @functools.wraps(f)
        def wrapped_f():
            params = signature(f).parameters
            args = [
                TegVar(f'{a}{i}') if v.annotation == TegVar else Var(f'{a}{i}')
                for i, (a, v) in enumerate(params.items())
            ]
            darg_names = [Sym(f'd{a}{i}') for i, a in enumerate(params)]
            e = Function(tuple(args), f(*args))
            e = simplify(e)

            ctx = {a.name: da_name for a, da_name in zip(args, darg_names)}
            de = deriv(e, ctx)
            de = simplify(de)
            return e, de

        return wrapped_f

    return potto_decorator


def list_to_param(params, vals):
    return {k.name: v for k, v in zip(params, vals)}


def eval_expr(e, ps, params):
    param_vals = list_to_param(params, ps)
    ret = evaluate_all(e, VarVal(param_vals), num_samples=5)
    print(f'eval: {ps}\n\t: {ret}')
    return ret


def eval_grad_expr(de, ps, params, dparams):
    grad = []
    for dparam in dparams[:1]:
        differential = {dp.name: 0 for dp in dparams}
        differential[dparam.name] = 1
        dparam_vals = list_to_param(params, ps) | differential
        dc_eval = evaluate_all(de, VarVal(dparam_vals), num_samples=20)
        grad.append(dc_eval)
    print(f'grad: {ps}\n\t: {grad}')
    return grad[0]
