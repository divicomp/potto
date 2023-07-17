import math
from numpy.random import uniform
from dataclasses import dataclass, field
from collections import defaultdict
from multiprocessing import Pool
from functools import partial
import numpy as np

from potto.ir.ir_env import (
    Const,
    UnaryBuiltin,
    Var,
    TegVar,
    Add,
    Mul,
    Div,
    Function,
    Int,
    IfElse,
    Delta,
    App,
    IREnv,
    Diffeomorphism,
)

from potto.lang.grammar import GExpr, Const as gConst
from potto.lang.samples import VarVal
from potto.lang.derivative import deriv, DerivDiffeo
from potto.lang.grammar import Sym
from potto.lang.evaluate_utils import (
    Environment,
    TraceEnv,
    GenSample,
    sample_arg_substitution,
    to_env,
    to_irenv,
    Gen,
    extend_samples_trace,
    multi_extend_samples_traces,
    flatten_args,
    get_full_sample_bundles,
    trim_trace,
)
from potto.lang.traces import TraceName


# NOTE: Idea: flip the ordering of the arrows and share common objects
# TODO: Relate each sample with its trace


def generate(expr, env, num_samples, gen_samples):
    """A single sample for every variable of integration in expr."""

    def gen(expr: IREnv, env: Environment, gen_samples: TraceEnv) -> Gen:
        match expr:
            case Const() | Var() | TegVar():
                return Gen()

            case UnaryBuiltin(e):
                unary_gen = gen(e, env, gen_samples)
                return extend_samples_trace(TraceName.Unary, unary_gen)

            case Add(left, right) | Mul(left, right) | Div(left, right):
                lgen = gen(left, env, gen_samples)
                rgen = gen(right, env, gen_samples)
                return multi_extend_samples_traces({TraceName.BinopLeft: lgen, TraceName.BinopRight: rgen})

            case App(function_, args, app_name) as expr:
                function: Function = evaluate(function_, env, gen_samples)

                # Evaluate returns Potto functions for functions
                nonflat_arg_vals = (evaluate(a, env, num_samples, gen_samples) for a in args)
                arg_vals = flatten_args(nonflat_arg_vals)
                assert len(function.arg_names) == len(arg_vals), f"{function.arg_names} {arg_vals} {function_}"

                arg_samps_pushdown, tvar_arg_name = sample_arg_substitution(function, args, arg_vals, gen_samples)

                # Add bounds for new variables of integration
                # e.g., x in [0, 1], y in [0, 2]
                # then (\w, z. 1)((x + 1, y))
                # w in [1, 2] and z in [0, 2]
                new_args = function.arg_names
                new_args_ind = 0
                for arg in args:
                    match arg:
                        case TegVar(sym):
                            new_name = new_args[new_args_ind].name
                            # TODO: this should only trigger for infinitesimals
                            # dx or dy don't need bounds
                            if sym in env.bounds:
                                env.bounds[new_name] = env.bounds[sym]
                            new_args_ind += 1
                        case Diffeomorphism(_, tvars, out_tvars) as diffeo:
                            if isinstance(diffeo.diffeo, DerivDiffeo):
                                new_args_ind += len(tvars)
                                continue
                            bounds = [env.get_bounds(tv.name) for tv in tvars]
                            lower_corner = tuple(gConst(lb) for lb, _ in bounds)
                            upper_corner = tuple(gConst(ub) for _, ub in bounds)
                            lbs, ubs = diffeo.bounds_transfer(lower_corner, upper_corner, env)
                            for otv, tvar, lb, ub in zip(out_tvars, tvars, lbs, ubs):
                                new_name = new_args[new_args_ind].name
                                env.bounds[new_name] = (lb, ub)
                                env.bounds[otv.name] = (lb, ub)
                                new_args_ind += 1
                        case _:
                            new_args_ind += 1

                # Update the environment with values from the application
                bindings = {n.name: v for n, v in zip(function.arg_names, arg_vals)}
                bindings[app_name] = function
                new_env = Environment(bindings, env)
                val_samps = TraceEnv(arg_samps_pushdown, gen_samples)
                body_gen = gen(function.body, new_env, val_samps)
                val_body_gen = body_gen | Gen({k: (v,) for k, v in arg_samps_pushdown.items()})

                # Reweight samples and collect traces from arguments
                good_weight_gen = reweight_samples_with_diffeo(
                    body_gen, val_body_gen, function, args, tvar_arg_name, env, gen_samples
                )
                all_gen = extend_samples_trace(TraceName.AppFun, good_weight_gen)
                for i, arg in enumerate(args):
                    all_gen |= extend_samples_trace(TraceName.AppArg, gen(arg, new_env, val_samps), i)
                return all_gen

            case Function():
                return Gen()

            case IfElse(cond, if_body, else_body):
                match cond:
                    case TegVar():
                        # TODO: Potential exponential blowup
                        # Static check on the bounds of integration?
                        if_gen = gen(if_body, env, gen_samples)
                        else_gen = gen(else_body, env, gen_samples)
                        return extend_samples_trace(TraceName.IfBody, if_gen) | extend_samples_trace(
                            TraceName.ElseBody, else_gen
                        )
                    case _:
                        cv = evaluate(cond, env, num_samples, gen_samples)
                        if cv > 0:
                            if_gen = gen(if_body, env, gen_samples)
                            return extend_samples_trace(TraceName.IfBody, if_gen)
                        else:
                            else_gen = gen(else_body, env, gen_samples)
                            return extend_samples_trace(TraceName.ElseBody, else_gen)

            case Delta(tvar, trace):
                return Gen({tvar.name: (GenSample(0, 1, trace),)})

            case Diffeomorphism():
                return Gen()

            case Int(integrand, measure):
                # Take a new sample and add it to the context
                sample_val = measure.sample(env, num_samples, gen_samples)
                sample = GenSample(sample_val, 1, None)

                # Add bounds to the environment
                lb = evaluate(measure.lower, env, num_samples, gen_samples)
                ub = evaluate(measure.upper, env, num_samples, gen_samples)
                bounds = {measure.tvar.name: (lb, ub)}
                new_env = Environment({measure.tvar.name: sample.sample}, env, bounds)

                # Account for the domain size
                # scale_gen = TraceEnv({measure.tvar.name: sample}, gen_samples)
                measure_size = naive_evaluate(measure.size(), new_env)
                sample = GenSample(sample_val, measure_size, None)
                new_gen_bindings = {measure.tvar.name: sample}
                if measure.dtvar:
                    new_gen_bindings[measure.dtvar.name] = GenSample(0, 1, None)
                new_gen = TraceEnv(new_gen_bindings, gen_samples)
                integrand_gen = gen(integrand, new_env, new_gen)
                next_gen = extend_samples_trace(TraceName.Integral, integrand_gen)
                return next_gen | Gen({measure.tvar.name: (sample,)})

            case _:
                raise TypeError(f"Unable to generate an object of type {type(expr)}")

    return gen(expr, env, gen_samples)


def naive_evaluate(expr: GExpr | IREnv, env: Environment):
    import potto.lang.grammar as g

    match expr:
        case Const(c):
            return c
        case Var(name):
            return env[name]
        case TegVar(name):
            return env[name]
        case UnaryBuiltin() as e:
            return e.eval(lambda e_, env_: naive_evaluate(e_, env_), env)
        case Add(left, right):
            return naive_evaluate(left, env) + naive_evaluate(right, env)
        case Mul(left, right):
            return naive_evaluate(left, env) * naive_evaluate(right, env)
        case Div(left, right):
            return naive_evaluate(left, env) / naive_evaluate(right, env)
        case Int():
            raise TypeError('Cannot naively evaluate a integral expression')
        case Delta():
            raise TypeError('Cannot naively evaluate a delta')
        case g.Const(c):
            return c
        case g.Var(name):
            return env[name]
        case g.TegVar(name):
            return env[name]
        case g.UnaryBuiltin() as e:
            return e.eval(lambda e_, env_: naive_evaluate(e_, env_), env)
        case g.Add(left, right):
            return naive_evaluate(left, env) + naive_evaluate(right, env)
        case g.Mul(left, right):
            return naive_evaluate(left, env) * naive_evaluate(right, env)
        case g.Div(left, right):
            return naive_evaluate(left, env) / naive_evaluate(right, env)
        case g.Int():
            raise TypeError('Cannot naively evaluate a integral expression')
        case g.Delta():
            raise TypeError('Cannot naively evaluate a delta')
        case _:
            raise TypeError(f'Cannot naively evaluate type {type(expr).__name__}')


def eval_integrand_once(
    i, expr, env, gen_samples, num_samples, integrand, new_env, new_gen, measure_lower, measure_upper, x
):
    trace_to_sample = {}
    all_samples = Gen()
    generated = generate(expr, env, num_samples, gen_samples).items()
    for tvar_name, samples in generated:
        new_samples = []
        for sample in samples:
            new_samples.append(sample)
        all_samples[tvar_name] = tuple(new_samples)

    for bundle in get_full_sample_bundles(all_samples):
        tvar_sample = bundle[x]
        if not measure_lower < tvar_sample.sample < measure_upper:  # Skip out-of-bounds samples
            continue
        weight = tvar_sample.weight if tvar_sample.trace is None else 1
        new_trace = bundle.trace.discard_first()[1] if bundle.trace is not None else None
        bundle.trace = new_trace
        bundle[x] = GenSample(tvar_sample.sample, tvar_sample.weight, None)
        bundle_gen = TraceEnv(bundle, new_gen, new_trace)
        bundle_env = Environment({t: v.sample for t, v in bundle.items()}, new_env)
        integrand_val = evaluate(integrand, bundle_env, num_samples, bundle_gen)
        trace_to_sample[bundle.trace] = weight * integrand_val
    return trace_to_sample


def evaluate(
    expr: IREnv | GExpr,
    env_or_var_val: Environment | VarVal | None = None,
    num_samples: int = 50,
    gen_samples: TraceEnv | None = None,
) -> float:
    expr = to_irenv(expr)
    env = to_env(env_or_var_val)
    gen_samples = TraceEnv() if gen_samples is None else gen_samples

    # TODO: we iterate through everything
    # but there should only be 1 tagged delta. Maybe reconsider data struct
    expr_trace, next_trace, gen_samples, none_gen_samples = trim_trace(gen_samples)
    trace_name = None if not expr_trace else expr_trace.name

    match expr:
        case Const(c):
            return c

        case Var(name):
            return env[name]

        case TegVar(name):
            return gen_samples[name].sample if name in gen_samples else env[name]

        case UnaryBuiltin():
            return expr.eval(lambda expr, env: evaluate(expr, env, num_samples, gen_samples), env)

        case Add(left, right):
            match trace_name:
                case TraceName.BinopLeft:
                    return evaluate(left, env, num_samples, gen_samples)
                case TraceName.BinopRight:
                    return evaluate(right, env, num_samples, gen_samples)
                case _:
                    return evaluate(left, env, num_samples, gen_samples) + evaluate(
                        right, env, num_samples, gen_samples
                    )

        case Mul(left, right):
            left_gen_samples, right_gen_samples = gen_samples, gen_samples
            match trace_name:
                case TraceName.BinopLeft:
                    right_gen_samples = none_gen_samples
                case TraceName.BinopRight:
                    left_gen_samples = none_gen_samples

            return evaluate(left, env, num_samples, left_gen_samples) * evaluate(
                right, env, num_samples, right_gen_samples
            )

        case Div(left, right):
            left_gen_samples, right_gen_samples = gen_samples, gen_samples
            match trace_name:
                case TraceName.BinopLeft:
                    right_gen_samples = none_gen_samples
                case TraceName.BinopRight:
                    left_gen_samples = none_gen_samples
            return evaluate(left, env, num_samples, left_gen_samples) / evaluate(
                right, env, num_samples, right_gen_samples
            )

        case Diffeomorphism(vars, tvars, _, _) as diffeo:
            return tuple(evaluate(expr, env, num_samples, gen_samples) for expr in diffeo.function(vars, tvars))

        case App(function_, args, name) as expr:
            body_trace = gen_samples.trace
            match trace_name:
                case TraceName.AppArg | None:
                    arg_gen_samples = gen_samples
                    body_trace = None
                case TraceName.AppFun:
                    arg_gen_samples = none_gen_samples
                case _:
                    raise ValueError(f'Cannot evaluate App expression with trace name {trace_name}')

            # Trace through arguments
            arg_vals = []
            for i, a in enumerate(args):
                if expr_trace is not None and i == expr_trace.arg_num:
                    arg_idx_gen_samples = arg_gen_samples
                else:
                    arg_idx_gen_samples = none_gen_samples

                arg_val = evaluate(a, env, num_samples, arg_idx_gen_samples)
                if isinstance(arg_val, tuple):
                    for s in arg_val:
                        arg_vals.append(s)
                else:
                    arg_vals.append(arg_val)

            # Evaluate the function to a Function and bind arguments
            function: Function = evaluate(function_, env, gen_samples)
            assert len(function.arg_names) == len(arg_vals)
            bindings = {n.name: v for n, v in zip(function.arg_names, arg_vals)}
            bindings[function.name] = function
            new_env = Environment(bindings, env)
            sample_bindings, tvar_arg_name = sample_arg_substitution(function, args, arg_vals, gen_samples)
            new_gen_samples = TraceEnv(sample_bindings, gen_samples, body_trace)
            return evaluate(function.body, new_env, num_samples, new_gen_samples)

        case Function():
            return expr

        case IfElse(cond, if_body, else_body):
            match trace_name:
                case TraceName.IfBody:
                    if evaluate(cond, env, num_samples, gen_samples) > 0:
                        return evaluate(if_body, env, num_samples, gen_samples)
                case TraceName.ElseBody:
                    if evaluate(cond, env, num_samples, gen_samples) <= 0:
                        return evaluate(else_body, env, num_samples, gen_samples)
                case _:
                    if evaluate(cond, env, num_samples, gen_samples) > 0:
                        return evaluate(if_body, env, num_samples, gen_samples)
                    else:
                        return evaluate(else_body, env, num_samples, gen_samples)
            return 0  # only happens when the delta produces a sample that triggers the other branch from the integral

        case Delta(_, trace):
            return 1 if trace_name is not None and trace_name == trace.name else 0

        case Int(integrand, measure) as expr:
            dx_binding = {measure.dtvar.name: GenSample(0, 1, None)} if measure.dtvar is not None else {}
            dx_binding_env = {measure.dtvar.name: 0} if measure.dtvar is not None else {}
            new_gen = TraceEnv(dx_binding, gen_samples, gen_samples.trace)
            new_env = Environment(dx_binding_env, env)

            # NOTE: maybe make this gen_samples.is_empty or similar in the future
            if not gen_samples.is_empty():
                integrand_val = evaluate(integrand, new_env, num_samples, new_gen)
                tvar_sample = gen_samples[measure.tvar.name]
                measure_lower, measure_upper = measure.get_bounds(env, num_samples, TraceEnv())
                if not measure_lower < tvar_sample.sample < measure_upper:
                    return 0
                weight = tvar_sample.weight
                if tvar_sample.trace is None:
                    weight *= measure.density(tvar_sample.sample, env, num_samples, gen_samples)
                return weight * integrand_val

            x = measure.tvar.name
            measure_lower, measure_upper = measure.get_bounds(env, num_samples, gen_samples)

            do_eval = partial(
                eval_integrand_once,
                expr=expr,
                env=env,
                gen_samples=gen_samples,
                num_samples=num_samples,
                integrand=integrand,
                new_env=new_env,
                new_gen=new_gen,
                measure_lower=measure_lower,
                measure_upper=measure_upper,
                x=x,
            )

            # if num_samples > 1:
            #     # parallel
            #     with Pool() as pool:
            #         monte_carlo_runs = pool.map(do_eval, range(num_samples))
            # else:
            # serial
            monte_carlo_runs = []
            for _ in range(num_samples):
                monte_carlo_runs.append(do_eval(_))

            trace_to_samples = defaultdict(list)
            for d in monte_carlo_runs:
                for k, v in d.items():
                    trace_to_samples[k].append(v)
            return sum(np.average(v) for v in trace_to_samples.values())

        case _:
            raise TypeError(f"Unable to generate an object of type {type(expr)}")


def evaluate_all(
    expr: IREnv | GExpr, env_or_var_val: Environment | VarVal | None = None, num_samples: int = 50
) -> float:
    expr = to_irenv(expr)
    env = to_env(env_or_var_val)

    return evaluate(expr, env, num_samples)


def reweight_samples_with_diffeo(body_gen, val_body_gen, function, args, tvar_arg_name, env, gen_samples):
    # Make the values of body_gen a list for efficiency and convenience
    body_gen_pullup_temp = defaultdict(list)
    for k, v in body_gen.items():
        body_gen_pullup_temp[k] = list(v)

    arg_ind = 0
    old_name_to_new = {}
    to_remove = set()
    for n, a in zip(function.arg_names, args):
        match a:
            case TegVar(name):
                body_gen_pullup_temp[name].extend(val_body_gen[n.name])
                to_remove.add(n.name)
                arg_ind += 1
            case Diffeomorphism(vars, tvars, out_tvars) as diffeo:
                arg_names = tuple(tvar_arg_name[tvar] for tvar in tvars)

                # For each variable, collect the samples from the integral and deltas
                delta_samples = defaultdict(dict)
                old_name_to_new |= dict(zip((otv.name for otv in tvars), (otv.name for otv in out_tvars)))

                # TODO: consider eliminating out_tvars
                for tvar in out_tvars + arg_names:
                    for sample in val_body_gen[tvar.name]:
                        if sample.trace is not None:
                            delta_samples[sample.trace][tvar.name] = sample

                for trace, ds_trace in delta_samples.items():
                    for i, (otv, tv, arg) in enumerate(zip(out_tvars, tvars, function.arg_names)):
                        sym = arg.name
                        if (
                            sym not in ds_trace
                            and isinstance(arg, TegVar)
                            and (arg_ind < function.infinitesimal_ind or function.infinitesimal_ind < 0)
                        ):
                            lb, ub = env.get_bounds(otv.name)

                            # Take a lebesgue sample from [lb, ub]
                            sample = uniform(lb, ub)
                            width = gen_samples[tv.name].weight
                            ds_trace[sym] = GenSample(sample, width, trace)
                            ds_trace[otv.name] = GenSample(sample, width, trace)

                # For each delta apply the inverse diffeomorphism and add that to the context
                # NOTE: we do the "dense" way of packing samples, which will take up more space
                # but is more convenient to work with.

                for complete_delta_sample in delta_samples.values():
                    samples = list(complete_delta_sample.values())
                    values_for_delta = {n: s.sample for n, s in complete_delta_sample.items()}
                    new_env = Environment(values_for_delta, env)
                    new_gen_samples = TraceEnv(complete_delta_sample, gen_samples)

                    inv_diffeo = diffeo.inverse(vars, arg_names)
                    new_samples = [
                        GenSample(
                            naive_evaluate(inv_fun, new_env),
                            samples[i].weight,
                            samples[0].trace,
                        )
                        for i, inv_fun in enumerate(inv_diffeo)
                    ]
                    env_bindings = {tv.name: s.sample for tv, s in zip(tvars, new_samples)}
                    nbindings = {tv.name: s for tv, s in zip(tvars, new_samples)}

                    n_env = Environment(env_bindings, new_env)
                    n_gen_samples = TraceEnv(nbindings, new_gen_samples)

                    # NOTE: Potential performance bug. We call out to derivative every iteration of eval
                    dtvar0 = Sym(f'd{diffeo.tvars[0].name.name}')
                    weight_dtvar0 = deriv(diffeo.weight, {diffeo.tvars[0].name: dtvar0})
                    weight = 1 / weight_dtvar0
                    env_for_weight = Environment({dtvar0: 1}, n_env)
                    w = abs(naive_evaluate(weight, env_for_weight))
                    new_samples[0] = GenSample(
                        new_samples[0].sample,
                        new_samples[0].weight * w,
                        new_samples[0].trace,
                    )
                    # Map each of the tvar names to their corresponding samples
                    for x, out_x, ns in zip(tvars, out_tvars, new_samples):
                        body_gen_pullup_temp[x.name].append(ns)
                        to_remove.add(out_x.name)

                    arg_ind += len(diffeo.tvars)
            case _:
                pass
    removed_abstraction = Gen({k: tuple(v) for k, v in body_gen_pullup_temp.items() if k not in to_remove})
    return removed_abstraction
