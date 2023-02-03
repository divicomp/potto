from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Union, ClassVar
from numbers import Number

from potto.lang.traces import Trace, TraceName


@dataclass(frozen=True)
class Uid:
    uid: int = field(default_factory=lambda: Uid.counter)
    counter: ClassVar[int] = 0

    def __post_init__(self):
        Uid.counter += 1

    def __str__(self):
        return f"{self.uid}"

    def __repr__(self):
        return f"{self.uid}"


@dataclass(frozen=True)
class Sym:
    name: str
    uid: Uid = field(default_factory=Uid)

    def __hash__(self) -> int:
        return self.uid.uid

    def __eq__(self, other) -> bool:
        if isinstance(other, Sym):
            return self.uid.uid == other.uid.uid and self.name == other.name
        return False

    def __str__(self) -> str:
        return f"{self.name}"

    def __repr__(self) -> str:
        return f"Sym({self.name}, {self.uid})"

    @classmethod
    def to_sym(cls, name: Union[str, "Sym"]):
        match name:
            case str(n):
                return Sym(n)
            case Sym():
                return name
            case _:
                raise ValueError(f"Cannot convert type {type(name).__name__} into Sym, for object: {name}")


@dataclass(frozen=True)
class GExpr(ABC):
    def __add__(self, other):
        return Add(self, GExpr.to_gexpr(other))

    def __radd__(self, other):
        return Add(GExpr.to_gexpr(other), self)

    def __sub__(self, other):
        return self + GExpr.to_gexpr(-other)

    def __rsub__(self, other):
        return GExpr.to_gexpr(other) + (Const(-1) * self)

    def __mul__(self, other):
        return Mul(self, GExpr.to_gexpr(other))

    def __rmul__(self, other):
        return Mul(GExpr.to_gexpr(other), self)

    def __truediv__(self, other):
        return Div(self, GExpr.to_gexpr(other))

    def __rtruediv__(self, other):
        return Div(GExpr.to_gexpr(other), self)

    def __neg__(self):
        return Const(-1) * self

    def __pow__(self, exp):
        exp_const: Const = GExpr.to_gexpr(exp)
        if exp_const.value == 0:
            return Const(1)
        # NOTE: this is naive linear, could be log. Also, should support more.
        assert isinstance(exp_const.value, int) and exp_const.value > 0, "We only support positive integer powers."
        return self * self ** (exp_const.value - 1)

    @staticmethod
    def to_gexpr(e):
        match e:
            case Number() as e:
                return Const(e)
            case GExpr():
                return e
            case _:
                raise ValueError(f"Cannot convert type {type(e).__name__} into GExpr, for object: {e}")


@dataclass(frozen=True)
class Var(GExpr):
    name: Sym = field()

    def __init__(self, name_impl: str | Sym):
        super().__init__()
        object.__setattr__(self, "name", Sym.to_sym(name_impl))

    def __str__(self):
        return f"{self.name}"


@dataclass(frozen=True)
class TegVar(GExpr):
    name: Sym

    def __init__(self, name_impl: str | Sym):
        super().__init__()
        object.__setattr__(self, "name", Sym.to_sym(name_impl))

    def __str__(self):
        return f"{self.name}"


@dataclass(frozen=True)
class Diffeomorphism(GExpr, ABC):
    vars: tuple[Var, ...]
    tvars: tuple[TegVar, ...]
    # TODO: remove depricated
    # NOTE: Defining weight does nothing
    # w = diffeo.function(diffeo.vars, diffeo.tvars)
    weight: GExpr | None = None  # 1 / |J_00 w(x)| (x_tilde)

    # TODO: Make this __call__ in the future.
    @abstractmethod
    def function(self, vars: tuple[GExpr, ...], tvars: tuple[GExpr, ...]) -> tuple[GExpr, ...]:
        pass

    @abstractmethod
    def inverse(self, vars: tuple[GExpr, ...], tvars: tuple[GExpr, ...]) -> tuple[GExpr, ...]:
        pass

    def __str__(self) -> str:
        return f"{tuple(str(i) for i in self.function(self.vars, self.tvars))}"


@dataclass(frozen=True)
class Const(GExpr):
    value: Number | float

    def __str__(self):
        return f"{self.value}"


@dataclass(frozen=True)
class IfElse(GExpr):
    cond: GExpr | Diffeomorphism
    if_body: GExpr
    else_body: GExpr
    trace: Trace = field(default_factory=lambda: Trace(name=TraceName.Leaf))

    def __str__(self):
        return f"{self.if_body} if ({self.cond}) else ({self.else_body})"


def Heaviside(cond: GExpr | Diffeomorphism):
    return IfElse(cond, Const(1.0), Const(0.0))


@dataclass(frozen=True)
class HeavisideNoDiff(GExpr):
    expr: GExpr

    def __str__(self):
        return f"HeavisideNoDiff(expr={self.expr})"


@dataclass(frozen=True)
class Delta(GExpr):
    expr: Diffeomorphism
    trace: Trace = field(default_factory=lambda: Trace(name=TraceName.Leaf))

    def __str__(self):
        return f"𝛿({str(self.expr)})"


@dataclass(frozen=True)
class Add(GExpr):
    left: GExpr
    right: GExpr

    def __str__(self):
        return f"{self.left} + {self.right}"


@dataclass(frozen=True)
class Mul(GExpr):
    left: GExpr
    right: GExpr

    def __str__(self):
        return f"({self.left}) * ({self.right})"


@dataclass(frozen=True)
class Div(GExpr):
    left: GExpr
    right: GExpr

    def __str__(self):
        return f"({self.left}) / ({self.right})"


@dataclass(frozen=True)
class Measure(ABC):
    lower: GExpr
    upper: GExpr
    tvar: TegVar
    dtvar: TegVar | None = None

    def __init__(self, lower, upper, tvar, dtvar=None):
        super().__init__()
        object.__setattr__(self, "lower", GExpr.to_gexpr(lower))
        object.__setattr__(self, "upper", GExpr.to_gexpr(upper))
        object.__setattr__(self, "tvar", tvar)
        object.__setattr__(self, "dtvar", dtvar)

    @abstractmethod
    def sample(self, env, num_samples, gen_samples):
        raise NotImplementedError

    @abstractmethod
    def get_bounds(self, env, num_samples, gen_samples) -> tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def density(self, x: float, env, num_samples, gen_samples):
        raise NotImplementedError

    @abstractmethod
    def size(self) -> GExpr:
        raise NotImplementedError

    def __str__(self) -> str:
        dtvar_str = f";{self.dtvar}" if self.dtvar else ""
        return f"{self.tvar}{dtvar_str} from {self.lower} to {self.upper} wrt {self.__class__.__name__}"


@dataclass(frozen=True)
class Int(GExpr):
    integrand: GExpr
    measure: Measure

    def __str__(self):
        return f"int {self.integrand} d{self.measure}"


@dataclass(frozen=True)
class Function(GExpr, ABC):
    arg_names: tuple[Var | TegVar, ...]
    body: GExpr
    name: Sym
    infinitesimal_ind: int

    def __init__(
        self, arg_names: tuple[Var | TegVar, ...], body: GExpr, name_impl: str | Sym = "", infinitesimal_ind: int = -1
    ):
        super().__init__()
        object.__setattr__(self, "arg_names", arg_names)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "name", Sym.to_sym(name_impl))
        object.__setattr__(self, "infinitesimal_ind", infinitesimal_ind)

    def __str__(self) -> str:
        return f"\\{self.arg_names}. {self.body}"


@dataclass(frozen=True)
class App(GExpr, ABC):
    function: GExpr
    args: tuple[GExpr]
    name: Sym

    def __init__(self, function: GExpr, args: tuple[GExpr, ...], name_impl: str | Sym = ""):
        super().__init__()
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "name", Sym.to_sym(name_impl))

    def __str__(self) -> str:
        return f"({self.function}) \n\t {tuple(str(a) for a in self.args)}"


@dataclass(frozen=True)
class UnaryBuiltin(GExpr):
    """A unary operator on Delta-free expressions such as Sqrt(x)."""

    expr: GExpr

    @abstractmethod
    def eval(self, evaluate, env) -> float:
        pass

    @abstractmethod
    def deriv(self, derivative, context) -> GExpr:
        pass


def heaviside(expr):
    return IfElse(expr, Const(1.0), Const(0.0))
