from numpy import var
from dataclasses import dataclass
from potto.lang.grammar import (
    Diffeomorphism,
    GExpr,
    Const,
    IfElse,
    UnaryBuiltin,
    Var,
    TegVar,
    Add,
    Mul,
    Div,
    Function,
    App,
    Int,
    Delta,
    Measure,
)
from potto.ir.ir_env import (
    IREnv,
    Const as IRConst,
    UnaryBuiltin as IRUnaryBuiltin,
    Add as IRAdd,
    Mul as IRMul,
    Div as IRDiv,
    Function as IRFunction,
    App as IRApp,
    Int as IRInt,
    Delta as IRDelta,
    TegVar as IRTegVar,
    Var as IRVar,
    Diffeomorphism as IRDiffeo,
    Measure as IRMeasure,
    IfElse as IRIfElse,
)


@dataclass(frozen=True)
class Meas(IRMeasure):
    mu: Measure = None

    def sample(self, env, num_samples, gen_samples):
        return self.mu.sample(env, num_samples, gen_samples)

    def get_bounds(self, env, num_samples, gen_samples):
        return self.mu.get_bounds(env, num_samples, gen_samples)

    def density(self, x: float, env, num_samples, gen_samples):
        return self.mu.density(x, env, num_samples, gen_samples)

    def size(self) -> IREnv:
        return grammar_to_ir_env(self.mu.size())


@dataclass(frozen=True)
class MyIRDiffeo(IRDiffeo):
    diffeo: Diffeomorphism

    def irvar_tvar_to_var_tvar(self, vars, tvars):
        return tuple(Var(v.name) for v in vars), tuple(TegVar(tv.name) for tv in tvars)

    def function(self, vars: tuple[IRVar, ...], tvars: tuple[IRTegVar, ...]) -> tuple[IREnv, ...]:
        vs, tvs = self.irvar_tvar_to_var_tvar(vars, tvars)
        return tuple(grammar_to_ir_env(e) for e in self.diffeo.function(vs, tvs))

    def inverse(self, vars: tuple[IRVar, ...], tvars: tuple[IRTegVar, ...]) -> tuple[IREnv, ...]:
        vs, tvs = self.irvar_tvar_to_var_tvar(vars, tvars)
        return tuple(grammar_to_ir_env(e) for e in self.diffeo.inverse(vs, tvs))

    def bounds_transfer(self, lower_left_corner: tuple[float], upper_right_corner: tuple[float], env):
        return self.diffeo.bounds_transfer(lower_left_corner, upper_right_corner, env)


@dataclass(frozen=True)
class UBuiltin(IRUnaryBuiltin):
    ubi: UnaryBuiltin

    def eval(self, evaluate, env) -> float:
        return self.ubi.eval(evaluate, env)

    def deriv(self, derivative, context) -> IREnv:
        return grammar_to_ir_env(self.ubi.deriv(derivative, context))


def grammar_to_ir_env(expr: GExpr) -> IREnv:
    match expr:

        case Const(value):
            return IRConst(value)

        case Var(name):
            return IRVar(name)

        case TegVar(name):
            return IRTegVar(name)

        case Add(left, right):
            irleft, irright = grammar_to_ir_env(left), grammar_to_ir_env(right)
            return IRAdd(irleft, irright)

        case Mul(left, right):
            irleft, irright = grammar_to_ir_env(left), grammar_to_ir_env(right)
            return IRMul(irleft, irright)

        case Div(left, right):
            irleft, irright = grammar_to_ir_env(left), grammar_to_ir_env(right)
            return IRDiv(irleft, irright)

        case Function(arg_names, body, name, infinitesimal_ind):
            irbody = grammar_to_ir_env(body)
            irarg_names = tuple(grammar_to_ir_env(a) for a in arg_names)
            return IRFunction(irarg_names, irbody, name, infinitesimal_ind)

        case App(function, args, name):
            irfunc = grammar_to_ir_env(function)
            irargs = tuple(grammar_to_ir_env(a) for a in args)
            return IRApp(irfunc, irargs, name)

        case Int(integrand, Measure(lower, upper, tvar, dtvar)) as expr:
            mu = expr.measure
            irlower = grammar_to_ir_env(lower)
            irupper = grammar_to_ir_env(upper)
            irtvar = grammar_to_ir_env(tvar)
            irdtvar = grammar_to_ir_env(dtvar) if dtvar is not None else None

            irmu = Meas(irlower, irupper, irtvar, irdtvar, mu)

            return IRInt(grammar_to_ir_env(integrand), irmu)

        case Diffeomorphism(vars, tvars) as diffeo:
            irvars = tuple(grammar_to_ir_env(v) for v in vars)
            irtvars = tuple(grammar_to_ir_env(tv) for tv in tvars)
            # We want to keep the original GExpr to take its derivative later
            # To get 1/ |dphi_00/dexpr|
            weight = diffeo.function(vars, tvars)[0]

            irout_tvars = tuple(IRTegVar(f'out_{tv.name.name}') for tv in tvars)
            return MyIRDiffeo(irvars, irtvars, irout_tvars, weight, diffeo)

        case IfElse(cond, if_body, else_body):
            match cond:
                case Diffeomorphism():
                    irdiffeo = grammar_to_ir_env(cond)
                    new_ifelse = IRIfElse(
                        irdiffeo.out_tvars[0], grammar_to_ir_env(if_body), grammar_to_ir_env(else_body)
                    )
                    return IRApp(IRFunction(irdiffeo.out_tvars, new_ifelse), (irdiffeo,))
                case _:
                    irifelse = IRIfElse(
                        grammar_to_ir_env(cond), grammar_to_ir_env(if_body), grammar_to_ir_env(else_body)
                    )
                    return irifelse

        case Delta(expr, trace):
            match expr:
                case Diffeomorphism() as diffeo:
                    irdiffeo = grammar_to_ir_env(diffeo)
                    new_delta = IRDelta(irdiffeo.out_tvars[0], trace)
                    return IRApp(IRFunction(irdiffeo.out_tvars, new_delta), (irdiffeo,))
                case TegVar() as tvar:
                    return IRDelta(grammar_to_ir_env(tvar), trace)
                case _:
                    raise Exception("Delta of a non-diffeo or tegvar")

        case UnaryBuiltin(e) as ubi:
            ire = grammar_to_ir_env(e)

            return UBuiltin(ire, ubi)

        case _:
            raise ValueError(f"Cannot convert expression {expr} of type {type(expr).__name__} to type IREnv. ")
