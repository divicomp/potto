from enum import Enum, auto

class TraceName(Enum):
    BinopLeft = auto()
    BinopRight = auto()
    AppArg = auto()
    AppFun = auto()
    Unary = auto()
    IfBody = auto()
    ElseBody = auto()
    Leaf = auto()
    Fun = auto()
    Integral = auto()


class Trace:
    def __init__(self, name: TraceName | None = None, next_trace=None, arg_num=None) -> None:
        self.name = name
        self.next_trace = next_trace
        self.arg_num = arg_num

    def __str__(self) -> str:
        return f'{[i for i in self]}'

    def __repr__(self) -> str:
        return f'{[i for i in self]}'

    @classmethod
    def add_trace(cls, name: TraceName, next_trace: 'Trace', arg_num=None):
        return None if next_trace is None else Trace(name, next_trace, arg_num)
    
    def discard_first(self) -> tuple:
        return self.name, self.next_trace

    def __hash__(self) -> int:
        return hash(tuple(i for i in self))

    def __eq__(self, other):
        return isinstance(other, Trace) and tuple(i for i in self) == tuple(i for i in other)

    def __iter__(self):
        if self.name is not None:
            yield self.name
            if self.next_trace is not None:
                yield from (i for i in self.next_trace)
