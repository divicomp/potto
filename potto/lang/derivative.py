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
    Int,
    Delta,
    Sym,
    App,
    Measure,
)
from potto.lang.samples import VarVal
from potto.lang.substitute import substitute


@dataclass(frozen=True)
class DerivMeas(Measure):
    mu: Measure = None

    def __init__(self, lower, upper, tvar, dtvar=None, mu=None):
        super().__init__(lower, upper, tvar, dtvar)
        object.__setattr__(self, "mu", mu)

    def sample(self, env, num_samples, gen_samples):
        return self.mu.sample(env, num_samples, gen_samples)

    def get_bounds(self, env, num_samples, gen_samples):
        return self.mu.get_bounds(env, num_samples, gen_samples)

    def density(self, x: float, env, num_samples, gen_samples):
        return self.mu.density(x, env, num_samples, gen_samples)

    def size(self):
        return self.mu.size()


@dataclass(frozen=True)
class DerivDiffeo(Diffeomorphism):
    diffeo: Diffeomorphism = None
    context: dict[Sym, Sym] = None

    def function(self, vars, tvars):
        var_sub = {v.name: vars[i] for i, v in enumerate(self.vars)}
        tvar_sub = {tv.name: tvars[i] for i, tv in enumerate(self.tvars)}
        dv = tuple(
            substitute(deriv(e, self.context), var_sub | tvar_sub)
            for e in self.diffeo.function(self.diffeo.vars, self.diffeo.tvars)
        )
        return dv

    def inverse(self, vars, tvars):
        var_sub = {v.name: vars[i] for i, v in enumerate(self.vars)}
        tvar_sub = {tv.name: tvars[i] for i, tv in enumerate(self.tvars)}
        dv = tuple(
            substitute(deriv(e, self.context), var_sub | tvar_sub)
            for e in self.diffeo.inverse(self.diffeo.vars, self.diffeo.tvars)
        )
        return dv


def deriv(expr: GExpr, context: dict[Sym, Sym]) -> GExpr:
    match expr:

        case Const():
            return Const(0)

        case Var(name):
            return Var(context[name]) if name in context else Const(0)

        case TegVar(name):
            return TegVar(context[name]) if name in context else Const(0)

        case Add(left, right):
            return deriv(left, context) + deriv(right, context)

        case Mul(left, right):
            dleft, dright = deriv(left, context), deriv(right, context)
            return dleft * right + left * dright

        case Div(left, right):
            dleft, dright = deriv(left, context), deriv(right, context)
            return (dleft * right - left * dright) / (right * right)

        case Function(arg_names, body, name):
            # (kmu) In the future, consider using a dual-like representation rather
            # than concatenating of all arguments with all infinitesimals.
            darg_names = tuple(type(n)(Sym(f"d{n}")) for n in arg_names)
            num_args = len(darg_names)
            new_arg_names = arg_names + darg_names
            new_context = context.copy()
            for n, dn in zip(arg_names, darg_names):
                new_context[n.name] = dn.name
            infinitesimal_ind = num_args if num_args > 0 else -1
            return Function(new_arg_names, deriv(body, new_context), name, infinitesimal_ind)

        case Diffeomorphism(vars, tvars) as diffeo:

            vars_infinitesimals = vars + tuple(Var(context[v.name]) for v in vars)
            tvars_infinitesimals = tuple(TegVar(context[tv.name]) for tv in tvars)

            weight = 1 / deriv(diffeo.function(vars, tvars)[0], {tvars[0]: 0})

            return DerivDiffeo(vars_infinitesimals, tvars_infinitesimals, weight, diffeo, context)

        case App(function, args, name):
            args_and_infinitesimals = args + tuple(deriv(a, context) for a in args)
            return App(deriv(function, context), args_and_infinitesimals, name)

        case Int(integrand, mu):
            new_context = context.copy()
            dx = Sym(f'd{mu.tvar.name.name}')
            new_context[mu.tvar.name] = dx
            lower_ctx = {mu.tvar.name: mu.lower, dx: Const(0)}
            upper_ctx = {mu.tvar.name: mu.upper, dx: Const(0)}
            # Leibniz Rule
            lower = (
                deriv(mu.lower, context) * substitute(integrand, lower_ctx)
                if isinstance(mu.lower, GExpr) and not isinstance(mu.lower, Const)
                else Const(0)
            )
            upper = (
                deriv(mu.upper, context) * substitute(integrand, upper_ctx)
                if isinstance(mu.upper, GExpr) and not isinstance(mu.upper, Const)
                else Const(0)
            )

            dmu = DerivMeas(mu.lower, mu.upper, mu.tvar, TegVar(dx), mu)
            return Int(deriv(integrand, new_context), dmu) + upper - lower

        case IfElse(cond, if_body, else_body, trace):
            interior_deriv = IfElse(cond, deriv(if_body, context), deriv(else_body, context))
            match cond:
                case Diffeomorphism(vars, tvars):
                    fout = cond.function(vars, tvars)[0]
                    return interior_deriv + Delta(cond, trace) * deriv(fout, context) * (if_body - else_body)
                case TegVar(name):
                    return interior_deriv + Delta(cond, trace) * deriv(cond, context) * (if_body - else_body)
                case _:
                    return interior_deriv

        case UnaryBuiltin() as expr:
            return expr.deriv(deriv, context)

        case _:
            raise ValueError(f"Derivatives not supported for {type(expr).__name__} type object: {expr}")
