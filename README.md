# Potto
Potto is a differentiable programming language that has an integral primitive and supports higher-order functions. This repository contains the core library implementation, while the applications can be found at https://github.com/divicomp/potto_applications. The applications include differentiating a ray tracing renderer.

## Installation Instructions
Potto requires Python 3.10+. To install Potto, run:
```
gh repo clone martinjm97/potto
cd potto
pip install -e .
```

## TODO: Illustrative Example


## Code structure
The `potto` folder contains the core implementation. We break the implementation into pieces:
- `ir` contains the code for an IR lowering that we do to abstract diffeomorphisms outside of Dirac deltas and conditionals, as well as add more information to diffeomorphisms to make evaluation easier.
- `lang` contains the core language implementation such as the grammar, evaluator, and derivative code.
- `libs` contains useful libraries for diffeomorphisms, measures, and math operators.

## Figures

Figure 2 is at `case_study/plot_normal.py`.

## TODO: Citation
