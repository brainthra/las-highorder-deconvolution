import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import ipywidgets as widgets
from IPython.display import display, HTML
import pickle
import os
import re
import json
import re
import argparse
from pathlib import Path

from lasim.spectra import *
from lasim.system import *
from lasim.utils import *

from laspectron.scalers import *
from laspectron.models import *
from laspectron.data import *
from laspectron.reconstructors import *
from laspectron.metrics import *
from laspectron.specgen import *

lyso = Scintillator('Lu2SiO5', 7.1, 25)
ch = Scintillator('C4H4', 1.6, 8)
bgo = Scintillator('Bi4Ge3O12', 7.1, 10)
yag = Scintillator('Y3Al5O12', 4.55, 35)

w = Filter('W', 19.1)
fe = Filter('Fe', 8.84)
al = Filter('Al', 2.7)

quartz = Filter('SiO2', 1.6)

mylar = Filter('C4H12O6', 1.6)
# endregion Imports

lim = "our"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--Bmin", type=float, required=True)
    parser.add_argument("--Bmax", type=float, required=True)
    parser.add_argument("--Bstep", type=float, required=True)
    parser.add_argument(
        "--Btype",
        required=True,
        type=lambda x: {
            "np.int64": np.int64,
            "float": float,
        }[x],
    )
    parser.add_argument("--Emin", type=float, required=True)
    parser.add_argument("--Emax", type=float, required=True)
    parser.add_argument("--target-bins", type=int, required=True)
    parser.add_argument("--ext-factor", type=float, required=True)

    parser.add_argument("--design", required=True)
    parser.add_argument("--spec-type", required=True)

    parser.add_argument("--noise-min", type=float, required=True)
    parser.add_argument("--noise-max", type=float, required=True)

    parser.add_argument("--flux-min", type=float, required=True)
    parser.add_argument("--flux-max", type=float, required=True)

    parser.add_argument("--model-type", required=True)
    parser.add_argument("--pred-bins-target", type=int, required=True)
    parser.add_argument("--pred-tar-scale", type=str, required=True)
    parser.add_argument("--pred-bins-ext", type=int, required=True)
    parser.add_argument("--pred-ext-scale", type=str, required=True)
    parser.add_argument("--ensembles", type=int, required=True)

    parser.add_argument("--seed", type=int, required=True)

    return parser.parse_args()

def build_path(
    root,
    design_name,
    Bmin, Bmax, Bstep, Btype,
    Emin, Emax, target_bins, ext_factor,
    design, spec_type,
    noise_min, noise_max,
    flux_min, flux_max,
    model_type,
    pred_bins_target, pred_tar_scale,
    pred_bins_ext, pred_ext_scale,
    ensembles,
    seed,
):
    name = (
        f"{Bmin}-{Bmax}-{Bstep}/"
        f"{Emin}-{Emax}--{target_bins}--{ext_factor}/"
        f"{design_name}/"
        f"{spec_type}/"
        f"noise--{noise_min}-{noise_max}/"
        f"flux--{flux_min}-{flux_max}/"
        f"{model_type}--{pred_bins_target}-{pred_tar_scale}--{pred_bins_ext}-{pred_ext_scale}--{ensembles}x/"
        f"{seed}"
    )

    return Path(root) / name


if __name__ == "__main__":

    args = parse_args()

    Bmin = args.Bmin
    Bmax = args.Bmax
    Bstep = args.Bstep
    Btype = args.Btype

    if Btype == np.int64:
        Bmin = int(args.Bmin)
        Bmax = int(args.Bmax)
        Bstep = int(args.Bstep)

    Emin = args.Emin
    Emax = args.Emax
    target_bins = args.target_bins
    ext_factor = args.ext_factor

    design = args.design
    design_name = design
    spec_type = args.spec_type

    noise_min = args.noise_min
    noise_max = args.noise_max

    flux_min = args.flux_min
    flux_max = args.flux_max

    model_type = args.model_type
    pred_bins_target = args.pred_bins_target
    pred_tar_scale = args.pred_tar_scale
    pred_bins_ext = args.pred_bins_ext
    pred_ext_scale = args.pred_ext_scale
    ensembles = args.ensembles

    seed = args.seed

    path = build_path(
        root="training",
        design_name=design_name,
        Bmin=Bmin, Bmax=Bmax, Bstep=Bstep, Btype=Btype,
        Emin=Emin, Emax=Emax, target_bins=target_bins, ext_factor=ext_factor,
        design=design, spec_type=spec_type,
        noise_min=noise_min, noise_max=noise_max,
        flux_min=flux_min, flux_max=flux_max,
        model_type=model_type,
        pred_bins_target=pred_bins_target, pred_tar_scale=pred_tar_scale,
        pred_bins_ext=pred_bins_ext, pred_ext_scale=pred_ext_scale,
        ensembles=ensembles,
        seed=seed
    )
    Path(path/"model").mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    target_range = (Emin, Emax)

    TARGET_ENERGIES = np.linspace(*target_range, target_bins)
    FULL_ENERGIES = np.linspace(target_range[1], target_range[1]*ext_factor, int(target_bins*(ext_factor-1.0))+1)
    FULL_ENERGIES = np.concatenate((TARGET_ENERGIES, FULL_ENERGIES[1:]))

    design:Spectrometer

    # Read and execute the file's content
    with open(f"./train-configs/designs/{design_name}.py", 'r') as f:
        code = f.read()

    exec(code)

    spectra_set:np.ndarray
    spectra_paramss: dict

    configs = json.load(open(f"./train-configs/specs/{spec_type}/{lim}.json", 'r'))
    with open(f"./train-configs/specs/{spec_type}/init.py", "r") as f:
        code = f.read()

    exec(code)

    target_idxs = np.linspace(0, target_bins-1, pred_bins_target).astype(int).tolist() if pred_tar_scale == 'lin' else np.geomspace(0.1, target_bins-1, pred_bins_target).astype(int).tolist()
    extended_idxs = np.geomspace(target_bins-1, int(target_bins*ext_factor)-1, pred_bins_ext+1).astype(int).tolist() if pred_ext_scale == 'log' else np.linspace(target_bins-1, int(target_bins*ext_factor)-1, pred_bins_ext+1).astype(int).tolist()

    sample_idxs = target_idxs + extended_idxs
    sample_idxs = np.array(sorted(list(set(sample_idxs))))

    ds = LASDataset(spectra_set, design, sample_idxs=sample_idxs)

    in_scaler = None
    out_scaler = get_scaler("minmax")

    out_scaler.fit(ds.spectra[:, sample_idxs])

    model:Model

    configs = json.load(open(f"./train-configs/models/{model_type}/{model_type}--{pred_bins_target}-{pred_tar_scale}--{pred_bins_ext}-{pred_ext_scale}--{ensembles}x/model.json", "r"))
    with open(f"./train-configs/models/{model_type}/init.py", "r") as f:
        code = f.read()
    exec(code)

    reconstructor = MultiReconstructor(design, model, ensembles, sample_idxs, target_range, in_scaler, out_scaler)

    digitiser = Camera(Bmin, Bmax, Bstep, Btype)
    optimal_flux = int(ds.get_optimal_flux(digitiser, up_offset=0.05))
    max_flux = optimal_flux * flux_max
    min_flux = optimal_flux * flux_min

    mutator = Mutator(digitiser, (min_flux, max_flux), (noise_min, noise_max), "linear", "linear", mature=0.0)

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    configs = json.load(open(f"./train-configs/models/{model_type}/{model_type}--{pred_bins_target}-{pred_tar_scale}--{pred_bins_ext}-{pred_ext_scale}--{ensembles}x/train.json", "r"))
    with open(f"./train-configs/models/{model_type}/train.py", "r") as f:
        code = f.read()
    exec(code)
    kwargs:dict

    train_ds, val_ds, test_ds = reconstructor.fit(ds, **kwargs)

    with open(f'{path}/model/{lim}.pkl', 'wb') as f:
            pickle.dump(reconstructor, f)

