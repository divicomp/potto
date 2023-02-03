from potto.lang.grammar import (
    GExpr,
    Const,
    IfElse,
    UnaryBuiltin,
    Var,
    TegVar,
    Add,
    Mul,
    Div,
    Int,
    Delta,
    Function,
    App,
)
from potto.lang.samples import VarVal
from potto.lang.evaluate import evaluate


def simplify(expr, var_val=None) -> GExpr:
    var_val = VarVal() if var_val is None else var_val
    match expr:

        case Const():
            return expr

        case Var(n):
            return Const(var_val[n]) if n in var_val else expr

        case TegVar():
            return expr

        case Add(left, right) | Mul(left, right) | Div(left, right):
            simplified_expr = type(expr)(simplify(left, var_val), simplify(right, var_val))
            match simplified_expr:
                case (Add(Const(), Const()) | Mul(Const(), Const()) | Div(Const(), Const())):  # Constant propagation
                    return Const(evaluate(simplified_expr))

                case Add(Const(0), e) | Mul(Const(1), e) | Add(e, Const(0)) | Mul(e, Const(1)) | Div(e, Const(1)):
                    return e

                case Mul(Const(0), _) | Mul(_, Const(0)) | Div(Const(0), _):
                    return Const(0)
                case (
                    Add(Add(e, Const(a)), Const(b))
                    | Add(Add(Const(a), e), Const(b))
                    | Add(Const(b), Add(e, Const(a)))
                    | Add(Const(b), Add(Const(a), e))
                ):
                    return Add(e, Const(a + b))
                case (
                    Mul(Mul(e, Const(a)), Const(b))
                    | Mul(Mul(Const(a), e), Const(b))
                    | Mul(Const(b), Mul(e, Const(a)))
                    | Mul(Const(b), Mul(Const(a), e))
                ):
                    return Mul(e, Const(a * b))
                # case (  TODO: distrubute rules?
                #     Mul(Add(e, Const(a)), Const(b)) |
                #     Mul(Add(Const(a), e), Const(b)) |
                #     Mul(Const(b), Add(e, Const(a))) |
                #     Mul(Const(b), Add(Const(a), e))
                # ):
                #     return Add(Mul(e, Const(b)), Const(a * b))
            return simplified_expr

        case Function(arg_names, body, name, infinitesimal_ind):
            return Function(arg_names, simplify(body, var_val), name, infinitesimal_ind)

        case App(function, args, name):
            return App(simplify(function, var_val), args, name)

        case Int(integrand, measure):
            return Int(simplify(integrand, var_val), measure)

        case IfElse(cond, if_body, else_body):
            ib = simplify(if_body, var_val)
            eb = simplify(else_body, var_val)
            if ib == eb == Const(0):
                return Const(0)
            else:
                return IfElse(cond, ib, eb)

        case Delta():
            return expr

        case UnaryBuiltin():
            return expr

        case _:
            raise ValueError(f"Expression of type {type(expr)} is not recognized.")
