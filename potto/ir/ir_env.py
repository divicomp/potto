from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Number

from potto.lang.grammar import Sym, TegVar, GExpr
from potto.lang.traces import Trace, TraceName


class IREnv:
    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (Const(-1) * self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __neg__(self):
        return Const(-1) * self


@dataclass(frozen=True)
class Support:
    lower: IREnv
    upper: IREnv


@dataclass(frozen=True)
class Var(IREnv):
    name: Sym

    def __init__(self, name_impl: str | Sym):
        super().__init__()
        object.__setattr__(self, "name", Sym.to_sym(name_impl))

    def __hash__(self) -> int:
        return self.name.__hash__()

    def __eq__(self, other) -> int:
        match other:
            case Var(name_other):
                return self.name == name_other
        return False

    def __str__(self):
        return f"{self.name}"


@dataclass(frozen=True)
class TegVar(IREnv):
    name: Sym

    def __init__(self, name_impl: str | Sym):
        super().__init__()
        object.__setattr__(self, "name", Sym.to_sym(name_impl))

    def __hash__(self) -> int:
        return self.name.__hash__()

    def __eq__(self, other) -> int:
        match other:
            case TegVar(name_other):
                return self.name == name_other
        return False

    def __str__(self):
        return f"{self.name}"


@dataclass(frozen=True)
class Function(IREnv):
    arg_names: tuple[Var | TegVar, ...]
    body: IREnv
    name: Sym
    infinitesimal_ind: int

    def __init__(
        self, arg_names: tuple[Var | TegVar, ...], body: IREnv, name_impl: str | Sym = "", infinitesimal_ind: int = -1
    ):
        super().__init__()
        object.__setattr__(self, "arg_names", arg_names)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "name", Sym.to_sym(name_impl))
        object.__setattr__(self, "infinitesimal_ind", infinitesimal_ind)

    def __str__(self) -> str:
        return f"\\{self.arg_names}. {self.body}"


@dataclass(frozen=True)
class Diffeomorphism(IREnv, ABC):
    vars: tuple[Var, ...]
    tvars: tuple[TegVar, ...]
    out_tvars: tuple[TegVar, ...]
    weight: GExpr

    # TODO: Make this __call__ in the future.
    @abstractmethod
    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[IREnv, ...]:
        pass

    @abstractmethod
    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[TegVar, ...]:
        pass

    def __str__(self) -> str:
        return f"{tuple(str(i) for i in self.function(self.vars, self.tvars))}"


@dataclass(frozen=True)
class Const(IREnv):
    value: Number | float

    def __str__(self):
        return f"{self.value}"


"""
@dataclass(frozen=True)
class Heaviside(IREnv):
    x: TegVar

    def __str__(self):
        return f"H({self.x})"
"""


@dataclass(frozen=True)
class IfElse(IREnv):
    cond: IREnv
    if_body: IREnv
    else_body: IREnv

    def __str__(self):
        return f"{self.if_body} if ({self.cond}) else ({self.else_body})"


@dataclass(frozen=True)
class Delta(IREnv):
    tvar: Var
    trace: Trace = field(default_factory=lambda: Trace(name=TraceName.Leaf))

    def __str__(self):
        return f"𝛿({self.tvar})"


@dataclass(frozen=True)
class Add(IREnv):
    left: IREnv
    right: IREnv

    def __str__(self):
        return f"{self.left} + {self.right}"


@dataclass(frozen=True)
class Mul(IREnv):
    left: IREnv
    right: IREnv

    def __str__(self):
        return f"({self.left}) * ({self.right})"


@dataclass(frozen=True)
class Div(IREnv):
    left: IREnv
    right: IREnv

    def __str__(self):
        return f"({self.left}) / ({self.right})"


@dataclass(frozen=True)
class Measure(ABC):
    lower: IREnv
    upper: IREnv
    tvar: TegVar
    dtvar: TegVar | None = None

    @abstractmethod
    def sample(self, env, num_samples, gen_samples):
        raise NotImplementedError

    @abstractmethod
    def get_bounds(self, env, num_samples, gen_samples):
        raise NotImplementedError

    @abstractmethod
    def density(self, x: float, env, num_samples, gen_samples):
        raise NotImplementedError

    @abstractmethod
    def size(self) -> IREnv:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.tvar} from {self.lower} to {self.upper} wrt {self.__class__.__name__}"


@dataclass(frozen=True)
class Int(IREnv):
    integrand: IREnv
    measure: Measure

    def __str__(self):
        return f"int {self.integrand} d{self.measure}"


@dataclass(frozen=True)
class App(IREnv, ABC):
    function: Function | Var
    args: tuple[IREnv]
    name: Sym

    def __init__(self, function: Function | Var, args, name_impl: str | Sym = ""):
        super().__init__()
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "name", Sym.to_sym(name_impl))

    def __str__(self) -> str:
        return f"({self.function}) \n\t {tuple(str(a) for a in self.args)}"


@dataclass(frozen=True)
class UnaryBuiltin(IREnv):
    """A unary operator on Delta-free expressions such as Sqrt(x)."""

    expr: IREnv

    @abstractmethod
    def eval(self, evaluate, env) -> float:
        pass

    @abstractmethod
    def deriv(self, derivative, context) -> IREnv:
        pass
