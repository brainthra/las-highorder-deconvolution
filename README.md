# lastesting

Trained models, results, input parameters and code for High-Order Spectral Deconvolution of Complex X-Ray Distributions Using Machine Learning

## Structure
```
.
|-- lasim/
|-- laspectron/
|-- training/
|-- testing/
|-- bench-our/
|-- bench-mpd/
|-- evaluation/
    |-- figs-bench.py
    |-- figs-iv.py
    |-- figs-group.py
    |-- fig-*.pdf
|-- train-all.py
|-- seed-train-script.py
|-- test-all.py
|-- seed-test-script.py
|-- our-bench-all.py
|-- seed-our-bench.py
|-- environment.yml
```

## Environment setup
Install using mamba from `environment.yml` file:
```bash
mamba env create -f environment.yml -n lastesting
mamba activate lastesting
```

## Installation
You need to install `MACR`, `lasim` and `laspectron`.
- `lasim` and `laspectron` are provided in this repository.
- `MACR` comes from [github](https://github.com/cda24/MACR) with the latest version used here pinned to a specific commit in `lasim`'s `pyproject.toml`.

Get to the directory of `laspectron` and run:
```bash
cd laspectron
pip install -e .
```
This would install all both `MACR` and `lasim` as well.


## Usage

To train the models, you can use the `train-all.py` script. It will train all the models and save the results in the `training` directory.

To run the tests, you can use the `test-all.py` script. It will run all the tests and generate the results in the `testing` directory. 

To run the benchmarks, you can use the `our-bench-all.py` script. This will run the benchmarks and generate the results in the `bench-our` directory.

To generate the figures, you can use the:
- `evaluation/figs-bench.py` script. It will generate the figures and save them in the `evaluation` directory.
- `evaluation/figs-iv.py` script. It will generate the figures and save them in the `evaluation` directory.
- `evaluation/figs-group.py` script. It will generate the figures and save them in the `evaluation` directory.