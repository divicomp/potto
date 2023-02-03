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

from potto.lang.evaluate_utils import to_irenv


def get_ast_size(expr):
    expr = to_irenv(expr)

    match expr:
        case Const(_):
            return 1

        case Var(_):
            return 1

        case TegVar(_):
            return 1

        case UnaryBuiltin():
            return 1

        case Add(left, right):
            return 1 + get_ast_size(left) + get_ast_size(right)

        case Mul(left, right):
            return 1 + get_ast_size(left) + get_ast_size(right)

        case Div(left, right):
            return 1 + get_ast_size(left) + get_ast_size(right)

        case Diffeomorphism(vars, tvars, _, _) as diffeo:
            size = 1
            for expr in diffeo.function(vars, tvars):
                size += get_ast_size(expr)
            return size

        case App(function_, args, name) as expr:
            size = 1
            for arg in args:
                size += get_ast_size(arg)
            size += get_ast_size(function_)
            return size

        case Function(args, body, name):
            size = 1
            for arg in args:
                size += get_ast_size(arg)
            size += get_ast_size(body)
            return size


        case IfElse(cond, if_body, else_body):
            return 1 + get_ast_size(cond) + get_ast_size(if_body) + get_ast_size(else_body)

        case Delta(tvar, _):
            return 1 + get_ast_size(tvar)

        case Int(integrand, measure) as expr:
            size = get_ast_size(integrand)
            size += get_ast_size(measure.lower)
            size += get_ast_size(measure.upper)
            size += 1
            return size

        case _:
            raise TypeError(f"Unable to get ast size of type {type(expr)}")
