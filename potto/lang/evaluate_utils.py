from dataclasses import dataclass, field
from collections import defaultdict

from potto.ir.ir_env import (
    TegVar,
    Function,
    IREnv,
    Diffeomorphism,
)
from potto.ir.ir_transform import grammar_to_ir_env
from potto.lang.grammar import GExpr, Sym
from potto.lang.samples import VarVal
from potto.lang.traces import Trace, TraceName


@dataclass(frozen=True)
class GenSample:
    sample: float
    weight: float
    trace: Trace | None


class Environment(dict):
    def __init__(self, bindings=None, parent=None, bounds=None):
        if not bindings:
            bindings = {}
        super().__init__(bindings if not parent else parent | bindings)
        self.parent = parent
        self.bounds = {} if bounds is None else bounds

    def get_bounds(self, tvar_name):
        if tvar_name in self.bounds:
            return self.bounds[tvar_name]
        elif self.parent is not None:
            return self.parent.get_bounds(tvar_name)
        raise KeyError(f"The variable {tvar_name} is not in the environment for bounds!")


class TraceEnv(Environment):
    def __init__(self, bindings=None, parent=None, trace: Trace | None = None):
        super().__init__(bindings, parent)
        self.trace = trace

    def is_empty(self):
        return len([i for i in self.keys()]) == 0


class Gen(dict):
    def __or__(self, other: "Gen") -> "Gen":
        new_gen = {}
        for var, samples in tuple(self.items()) + tuple(other.items()):
            if var in new_gen:
                new_gen[var] += tuple(samples)
            else:
                new_gen[var] = tuple(samples)
        return Gen(new_gen)


def to_irenv(expr: GExpr | IREnv | None) -> IREnv:
    match expr:
        case GExpr():
            expr = grammar_to_ir_env(expr)
            return expr
        case IREnv():
            return expr
        case _:
            raise ValueError("The given expression should be either a GExpr or an IREnv.")


def to_env(env_varval_none: Environment | VarVal | None) -> Environment:
    match env_varval_none:
        case Environment():
            env = env_varval_none
        case VarVal(d):  # Unpack a VarVal into a new environment
            env = Environment()
            for k, v in d.items():
                env[k] = v
        case None:
            env = Environment()
        case _:
            raise NotImplementedError
    return env


def sample_arg_substitution(
        function: Function, args: tuple[IREnv, ...], arg_vals: tuple[float, ...], gen_samples: TraceEnv
):
    arg_samps = {}
    ind_to_sample = {}

    def flat_args(_args):
        """Unpack variables of integration."""
        for t in _args:
            match t:
                case TegVar() as arg:
                    yield (arg, None)
                case Diffeomorphism(_, tvars) as arg:
                    for i in range(len(tvars)):
                        yield (arg, i)
                case _:
                    yield (t, None)

    for i, (n, (a, aidx), v) in enumerate(zip(function.arg_names, flat_args(args), arg_vals)):
        match n:
            case TegVar(arg_name):
                match a:
                    case TegVar(val_name):  # TODO: generalize when arg val is not a TegVar?
                        old_samp = gen_samples[val_name]
                        sample = GenSample(v, old_samp.weight, old_samp.trace)
                        arg_samps[arg_name] = sample
                        ind_to_sample[i] = (a, sample)
                    case Diffeomorphism(vars, tvars, diffeo_out_tvars, weight) as a:
                        in_tvar = tvars[aidx]
                        out_tvar = diffeo_out_tvars[aidx]

                        prev_sample = gen_samples[in_tvar.name]
                        sample = GenSample(v, prev_sample.weight, prev_sample.trace)
                        arg_samps[out_tvar.name] = sample
                        ind_to_sample[i] = (in_tvar, sample)

    tvar_to_arg_name = {}
    for i, (in_tvar, sample) in ind_to_sample.items():
        arg_samps[function.arg_names[i].name] = sample
        tvar_to_arg_name[in_tvar] = function.arg_names[i]

    return arg_samps, tvar_to_arg_name


def extend_env_trace(arg_names, name: TraceName, gen_samples: Environment) -> Environment:
    return Environment(
        {
            var_name: GenSample(sample.sample, sample.weight, Trace.add_trace(name, sample.trace))
            for var_name in arg_names
            for sample in gen_samples[var_name]
        },
        gen_samples.parent,
    )


def extend_samples_trace(name: TraceName, samples: Gen, arg_num=None) -> Gen:
    name_samples = defaultdict(list)
    for var_name, samples in samples.items():
        for sample in samples:
            new_trace = Trace.add_trace(name, sample.trace, arg_num)
            name_samples[var_name].append(GenSample(sample.sample, sample.weight, new_trace))

    return Gen({k: v for k, v in name_samples.items()})


def multi_extend_samples_traces(name_samples: dict[TraceName, Gen]) -> Gen:
    dicts = [dict(extend_samples_trace(name, samples).items()) for name, samples in name_samples.items()]
    x = {}
    for d in dicts:
        for k, v in d.items():
            if k not in x:
                x[k] = v
            else:
                x[k] += v
    return Gen(x)
    # TODO: why doesn't this work?
    # gens = Gen()
    # for name, samples in name_samples.items():
    #     ext = extend_samples_trace(name, samples)
    #     gens |= ext
    # return gens


def flatten_args(nonflat_arg_vals):
    # Unpack tuple arguments, which is relevant for multidimensional diffeos
    arg_vals = []
    for a in nonflat_arg_vals:
        if isinstance(a, tuple):
            arg_vals.extend(a)
        else:
            arg_vals.append(a)
    return arg_vals


def trim_trace(gen_samples: TraceEnv):
    orig_trace = gen_samples.trace
    next_trace = None
    none_gen_samples = gen_samples
    new_gen_samples = gen_samples

    if gen_samples.trace is not None:
        _, next_trace = gen_samples.trace.discard_first()
        new_gen_samples = TraceEnv({}, gen_samples, next_trace)
        none_gen_samples = TraceEnv({}, gen_samples)
    return orig_trace, next_trace, new_gen_samples, none_gen_samples


@dataclass
class SampleBundle(dict[Sym, GenSample]):
    trace: Trace | None = field(default=None)


def get_full_sample_bundles(g: Gen) -> list[SampleBundle]:
    trace_to_sample_bundle: dict[Trace | None, SampleBundle] = defaultdict(SampleBundle)
    for tv, samps in g.items():
        for s in samps:
            bundle = trace_to_sample_bundle[s.trace]
            bundle[tv] = GenSample(s.sample, s.weight, None)
            bundle.trace = s.trace
    base_bundle = trace_to_sample_bundle[None]
    all_full_bundles = []
    for bundle in trace_to_sample_bundle.values():
        unset_tvs = base_bundle.keys() - bundle.keys()
        for tv in unset_tvs:
            bundle[tv] = base_bundle[tv]
        all_full_bundles.append(bundle)
    return all_full_bundles
