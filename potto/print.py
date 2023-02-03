from typing import Any, Callable
from functools import wraps

from potto.lang.grammar import (
    Int,
    Const,
    TegVar,
    Var,
    Delta,
    Measure,
    Sym,
    Diffeomorphism,
    GExpr,
    Mul,
    Add,
    Div,
    Function,
    App,
    IfElse,
    HeavisideNoDiff,
    UnaryBuiltin,
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
    Var as IRVar,
    Diffeomorphism as IRDiffeo,
    Measure as IRMeasure,
    IfElse as IRIfElse,
    TegVar as IRTegVar,
)

from potto.lang.traces import Trace
from potto.libs.measure import BoundedLebesgue


class PrintStream(object):
    """
    Prints to console
    # TODO: refactor subclasses explicitly as token streams?
    """

    def __init__(self):
        pass

    def print(self, *args: Any):  # TODO: have print take in outfile flag instead of separate FilePrintStream?
        for v in args:
            match v:
                case str(s):
                    print(s, end='')
                case _:
                    raise ValueError(f'Cannot print_stream an object of type {type(v).__name__}')

    def println(self, *args: Any):
        self.print(*args)
        print('', end='\n')

    def __enter__(self) -> 'PrintStream':
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass


class FilePrintStream(PrintStream):
    """
    Prints to a file
    """

    def __init__(self, filename: str, append=False):
        super().__init__()
        self.filename = filename
        if not append:
            with open(self.filename, 'w') as f:
                pass

    def print(self, *args: Any):
        with open(self.filename, 'a') as f:
            for v in args:
                match v:
                    case str(s):
                        print(s, end='')
                        print(s, end='', file=f)
                    case _:
                        raise ValueError(f'Cannot print_stream an object of type {type(v).__name__}')

    def println(self, *args: Any):
        self.print(*args)
        with open(self.filename, 'a') as f:
            print('', end='\n')
            print('', end='\n', file=f)


class CompactPrintStream(PrintStream):
    """
    Suppresses empty newlines
    # TODO: make this work with indents?
    # TODO: have this functionality work by default?
    """

    def __init__(self, o: PrintStream):
        super().__init__()
        self.o = o
        self.empty_line = True

    def print(self, *args: Any):
        def is_whitespace(v):
            return isinstance(v, str) and v.isspace()

        if not all(map(is_whitespace, args)):
            self.empty_line = False
        self.o.print(*args)

    def println(self, *args: Any):
        self.print(*args)
        if not self.empty_line:
            self.o.println()
        self.empty_line = True


class ObjectPrintStream(PrintStream):
    """
    Prints non-string objects via the supplied print function
    # TODO: currently only works for 1 print_func at a time when in conjunction with IndentBlock
    """

    def __init__(self, o: PrintStream, print_func: Callable):
        super().__init__()
        self.o = o
        self.print_func = print_func

    def print(self, *args: Any):
        for v in args:
            match v:
                case str(s):
                    self.o.print(s)
                case _:
                    self.print_func(self.o, v)

    def println(self, *args: Any):
        self.print(*args)
        self.o.println()


class IndentStream(PrintStream):
    """
    Adds indentation to prints
    """

    def __init__(self, o: PrintStream, num_indents: int = 1, indent_size: int = 4):
        super().__init__()
        self.o = o
        self.num_indents = num_indents
        self.indent_length = indent_size
        self.just_printlned = True

    def print(self, *args: Any):
        if args and self.just_printlned:
            self.o.print(' ' * (self.indent_length * self.num_indents))
        self.o.print(*args)
        self.just_printlned = False

    def println(self, *args: Any):
        if args and self.just_printlned:  # TODO: double check (if args) condition?
            self.o.print(' ' * (self.indent_length * self.num_indents))
        self.o.println(*args)
        self.just_printlned = True


class IndentBlock(object):
    """
    Context manager to add indents to object printing
    # TODO: fold this into PrintStream subclass behavior?
    """

    def __init__(self, o: ObjectPrintStream, num_indents: int = 1, indent_size: int = 4) -> None:
        self.o = o
        self.num_indents = num_indents
        self.indent_size = indent_size

    def __enter__(self) -> ObjectPrintStream:
        return ObjectPrintStream(IndentStream(self.o.o, self.num_indents, self.indent_size), self.o.print_func)

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass


def object_printer(p: Callable):
    """
    Allows p to use itself as the print function for object printing
    """

    @wraps(p)
    def oprint(o: PrintStream, *args):
        o = ObjectPrintStream(o, oprint)
        return p(o, *args)

    return oprint


@object_printer
def print_expr(o: ObjectPrintStream, e: GExpr | Sym):
    match e:
        case tuple() as e:
            o.print(*e)
        case Sym(name, uid):
            o.print(f'{name}@{uid.uid}')
        case Const(c):
            o.print(f'{c}')
        case Var(name) | TegVar(name):
            o.print(name)
        case Add(left, right) as e:
            o.print(left, f' + ', right)
        case Mul(left, right) as e:
            o.print(left, f' * ', right)
        case Div(left, right) as e:
            o.print(left, f' / ', right)
        case UnaryBuiltin(expr):
            # o.print(f'{name} $ ', expr)
            raise NotImplementedError
        case Function(arg_names, body, name):
            o.println('Function ', name, ':')
            with IndentBlock(o) as o1:
                o1.println('Args:')
                with IndentBlock(o1) as o2:
                    for i, n in enumerate(arg_names):
                        o2.print(' ' if i > 0 else '', n)
                    o2.println('.')
                o1.println('Body:')
                with IndentBlock(o1) as o2:
                    o2.println(body)
        case App(function, arg_vals, name):
            o.println('App ', name, ':')
            with IndentBlock(o) as o1:
                o1.println('Arg_vals:')
                with IndentBlock(o1) as o2:
                    for i, n in enumerate(arg_vals):
                        o2.println(f'{i}: ', n)
                o1.print('Func: ')
                o1.println(function)
        case Measure(lower, upper, tvar) as e:
            # right now put measure bounds on all one line
            # TODO: add fstring support for non-newline args? just use str method naively?
            o.print(tvar, ' from ', lower, ' to ', upper, ' wrt ', type(e).__name__, ':')
        case Int(integrand, measure):
            o.println('Int ', measure)
            with IndentBlock(o) as o1:
                o1.println(integrand)
        case Delta(expr, trace):
            o.println(f'𝛿({expr}, {trace})')
        case Diffeomorphism(vars, tvars, weight) as diffeo:
            o.println('Diffeo(', vars, ',', tvars, ',', weight)
            with IndentBlock(o) as o1:
                o1.println(diffeo.function(vars, tvars))
        case HeavisideNoDiff(expr):
            o.print(expr, ' > 0')
        case IfElse(cond, if_body, else_body):
            o.println('If ', cond, ':')
            with IndentBlock(o) as o1:
                o1.println(if_body)
            o.println('Else:')
            with IndentBlock(o) as o1:
                o1.println(else_body)

        case IRConst(c):
            o.print(f'{c}')
        case IRVar(name) | IRTegVar(name):
            o.print(name)
        case IRAdd(left, right) as e:
            o.print(left, f' + ', right)
        case IRMul(left, right) as e:
            o.print(left, f' * ', right)
        case IRDiv(left, right) as e:
            o.print(left, f' / ', right)
        case IRUnaryBuiltin(expr):
            # o.print(f'{name} $ ', expr)
            raise NotImplementedError
        case IRFunction(arg_names, body, name):
            o.println('Function ', name, ':')
            with IndentBlock(o) as o1:
                o1.println('Args:')
                with IndentBlock(o1) as o2:
                    for i, n in enumerate(arg_names):
                        o2.print(' ' if i > 0 else '', n)
                    o2.println('.')
                o1.println('Body:')
                with IndentBlock(o1) as o2:
                    o2.println(body)
        case IRApp(function, arg_vals, name):
            o.println('App ', name, ':')
            with IndentBlock(o) as o1:
                o1.println('Arg_vals:')
                with IndentBlock(o1) as o2:
                    for i, n in enumerate(arg_vals):
                        o2.println(f'{i}: ', n)
                o1.print('Func: ')
                o1.println(function)
        case IRMeasure(lower, upper, tvar) as e:
            # right now put measure bounds on all one line
            # TODO: add fstring support for non-newline args? just use str method naively?
            o.print(tvar, ' from ', lower, ' to ', upper, ' wrt ', type(e).__name__, ':')
        case IRInt(integrand, measure):
            o.println('Int ', measure)
            with IndentBlock(o) as o1:
                o1.println(integrand)
        case IRDelta(x, trace):
            o.println('delta(', x, ', ', trace, ')')
        case IRDiffeo(vars, tvars, weight) as diffeo:
            o.println('Diffeo(', vars, ',', tvars, ',', weight)
            with IndentBlock(o) as o1:
                o1.println(diffeo.function(vars, tvars))
        case IRIfElse(cond, if_body, else_body):
            o.println('If ', cond, ':')
            with IndentBlock(o) as o1:
                o1.println(if_body)
            o.println('Else:')
            with IndentBlock(o) as o1:
                o1.println(else_body)
        case Trace() | None as t:
            o.println('trace', str(t))
        case _:
            raise ValueError(f'Cannot print object of type {type(e).__name__}')


def get_printer(f: Callable, out_file: str | None = None):
    base_o = FilePrintStream(out_file) if out_file else PrintStream()
    return ObjectPrintStream(CompactPrintStream(base_o), f)


def main():
    # o = get_printer(print_expr, out_file='../out/log.txt')
    o = get_printer(print_expr)
    o.println(Sym('h'))
    o.println(Var('x'))
    o.println(Sym('h'))

    t = Var('t')
    x = TegVar('x')
    f_body = IfElse(HeavisideNoDiff(t + 7), t * t + x * 2, Const(0))
    f = Function((t, x), f_body, Sym('f'))
    o.println(f)
    o.println('h')
    tval = Var('tval')
    xval = TegVar('xval')
    a = App(f, (tval * 4, xval))
    mu = BoundedLebesgue(0, 1, x)
    i = Int(a, mu)
    o.println(a + i * 2)

    n0 = Var('n0')
    f0 = Var('f0')

    t = Var('t')
    x = TegVar('x')

    t_ = Var('t_')
    comb_fiber = App(Function((t_,), x / 10 * 12), (t + n0,))
    fact_recurse = comb_fiber + App(f0, (f0, n0 - 1)) + 3
    fact_default = Const(0)
    fact_body = IfElse(n0, fact_recurse, fact_default)
    fact_func = Function((f0, n0), fact_body, Sym('fact'))

    def potto_fix(f: Function):
        ret_args = f.arg_names[1:]
        ret_body = App(f, (f, *ret_args))
        ret_name = Sym(f'fix_{f.name.name}')
        return Function(ret_args, ret_body, ret_name)

    fix_fact = potto_fix(fact_func)

    n = Var('n')
    integrand = App(fix_fact + 2, (n,))

    mu = BoundedLebesgue(Const(0), Const(10), x)
    expr = Int(integrand, mu)
    print(expr)
    o.print(expr)


if __name__ == '__main__':
    main()
