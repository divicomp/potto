from potto.lang.grammar import (
    Diffeomorphism,
    GExpr,
    Const,
    UnaryBuiltin,
    Var,
    TegVar,
    Add,
    Mul,
    Div,
    Int,
    Sym,
    IfElse,
    App,
    Function,
)


def substitute(expr: GExpr, context: dict[Sym, GExpr]) -> GExpr:
    match expr:

        case Const() | Var():
            return expr

        case TegVar(name):
            return context[name] if name in context else expr

        case Add(left, right) | Mul(left, right) | Div(left, right):
            return type(expr)(substitute(left, context), substitute(right, context))

        case Int(integrand, mu):
            lower, upper = mu.lower, mu.upper
            if isinstance(mu.lower, GExpr):
                lower = substitute(mu.lower, context)
            if isinstance(mu.upper, GExpr):
                upper = substitute(mu.upper, context)

            mu = mu.__class__(lower, upper, mu.tvar)
            integrand = substitute(integrand, context)
            return Int(integrand, mu)

        case IfElse(cond, if_body, else_body):
            sub_if, sub_else = substitute(if_body, context), substitute(else_body, context)
            match cond:
                case Diffeomorphism(vars, tvars):
                    sub_cond = substitute(cond.function(vars, tvars)[0], context)
                case _:
                    sub_cond = substitute(cond, context)
            return IfElse(sub_cond, sub_if, sub_else)

        case UnaryBuiltin(e):
            return type(expr)(substitute(e, context))

        case App(fun, args):
            new_args = tuple(substitute(arg, context) for arg in args)
            new_fun = substitute(fun, context)
            return App(new_fun, new_args)

        case Function(arg_names, body):
            assert not any(
                an.name in context for an in arg_names), "name conflict. In the future we should resolve this"
            return Function(arg_names, substitute(body, context))

        case _:
            raise TypeError(f"Unable to evaluate an object of type {type(expr)}")
