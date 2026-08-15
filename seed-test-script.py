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
        root="testing",
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
    Path(path/"res").mkdir(parents=True, exist_ok=True)

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

    digitiser = Camera(Bmin, Bmax, Bstep, Btype)
    optimal_flux = int(ds.get_optimal_flux(digitiser, up_offset=0.05))
    max_flux = optimal_flux * flux_max
    min_flux = optimal_flux * flux_min

    curdir = str(path).replace("testing", "training")
    reconstructor:MultiReconstructor = pickle.load(open(f'{curdir}/model/{lim}.pkl', 'rb'))
    train_ds, val_ds, test_ds = ds.split([0.6, 0.15, 0.25], shuffle=False)
    train_ds.prepare(reconstructor.out_scaler)
    val_ds.prepare(reconstructor.out_scaler)
    test_ds.prepare(reconstructor.out_scaler)

    from matplotlib.colors import LogNorm

    def evaluate_reconstructions_on_testset(flux_min=0.01, flux_max=1.0,
                                            min_noise=0.0, max_noise=1.0,
                                            digitiser:Camera=digitiser):
        min_flux = optimal_flux * flux_min
        max_flux = optimal_flux * flux_max

        mutator = Mutator(digitiser, (min_flux, max_flux), (min_noise, max_noise), "linear", "linear")
        np.random.seed(seed)
        modded_ms, fluxes = mutator.prepare(test_ds.measureds, (min_flux, max_flux), (min_noise, max_noise), return_flux=True)
        modded_ss = test_ds.spectra * fluxes[:, np.newaxis]

        recon_infos = [reconstructor.inference(modded_ms[idx], method="opt", digitiser=digitiser, return_info=True, debug=False)[1] for idx in range(len(test_ds))]
        recon_preds = [info.spec_tar for info in recon_infos]
        recon_ms = [digitiser.digitise(reconstructor.spectrometer(info.spec_full)) for info in recon_infos]
        recon_fluxes = np.array([info.flux for info in recon_infos])
        recon_djss = shapeErrors(recon_preds, (modded_ss)[:, :target_bins]).target
        recon_maes = mae(recon_preds, (modded_ss)[:, :target_bins])/fluxes
        recon_fae = rae(recon_fluxes, fluxes)
        recon_mrs = np.array(recon_ms) - modded_ms
        recon_mrp = recon_mrs / modded_ms
        np.save(f'{path}/res/djs--B-{digitiser.visible}-{digitiser.saturation}-{digitiser.step}--flux-{flux_min}-{flux_max}--noise-{min_noise}-{max_noise}--{lim}.npy', recon_djss)
        np.save(f'{path}/res/mae--B-{digitiser.visible}-{digitiser.saturation}-{digitiser.step}--flux-{flux_min}-{flux_max}--noise-{min_noise}-{max_noise}--{lim}.npy', recon_maes)
        np.save(f'{path}/res/fae--B-{digitiser.visible}-{digitiser.saturation}-{digitiser.step}--flux-{flux_min}-{flux_max}--noise-{min_noise}-{max_noise}--{lim}.npy', recon_fae)
        np.save(f'{path}/res/flx--B-{digitiser.visible}-{digitiser.saturation}-{digitiser.step}--flux-{flux_min}-{flux_max}--noise-{min_noise}-{max_noise}--{lim}.npy', recon_fluxes)
        np.save(f'{path}/res/mrs--B-{digitiser.visible}-{digitiser.saturation}-{digitiser.step}--flux-{flux_min}-{flux_max}--noise-{min_noise}-{max_noise}--{lim}.npy', recon_mrs)
        np.save(f'{path}/res/mrp--B-{digitiser.visible}-{digitiser.saturation}-{digitiser.step}--flux-{flux_min}-{flux_max}--noise-{min_noise}-{max_noise}--{lim}.npy', recon_mrp)


    test_digitiser = Camera(10, 65535, 1, np.int64)
    evaluate_reconstructions_on_testset(flux_min=0.01, flux_max=1.0,
                                            min_noise=0.0, max_noise=0.0,
                                            digitiser=test_digitiser)
    
    test_digitiser = Camera(10, 65535, 1, np.int64)
    evaluate_reconstructions_on_testset(flux_min=0.01, flux_max=1.0,
                                            min_noise=0.0, max_noise=1.0,
                                            digitiser=test_digitiser)
    
    test_digitiser = Camera(10, 65535, 1, np.int64)
    evaluate_reconstructions_on_testset(flux_min=1.0, flux_max=1.0,
                                            min_noise=0.0, max_noise=0.0,
                                            digitiser=test_digitiser)
    test_digitiser = Camera(10, 65535, 1, np.int64)
    evaluate_reconstructions_on_testset(flux_min=1.0, flux_max=1.0,
                                            min_noise=0.0, max_noise=1.0,
                                            digitiser=test_digitiser)

    test_digitiser = Camera(10, 65535, 1, np.int64)
    evaluate_reconstructions_on_testset(flux_min=0.1, flux_max=0.1,
                                            min_noise=0.0, max_noise=0.0,
                                            digitiser=test_digitiser)
    test_digitiser = Camera(10, 65535, 1, np.int64)
    evaluate_reconstructions_on_testset(flux_min=0.1, flux_max=0.1,
                                            min_noise=0.0, max_noise=1.0,
                                            digitiser=test_digitiser)

    test_digitiser = Camera(10, 65535, 1, np.int64)
    evaluate_reconstructions_on_testset(flux_min=0.01, flux_max=0.01,
                                            min_noise=0.0, max_noise=0.0,
                                            digitiser=test_digitiser)
    test_digitiser = Camera(10, 65535, 1, np.int64)
    evaluate_reconstructions_on_testset(flux_min=0.01, flux_max=0.01,
                                            min_noise=0.0, max_noise=1.0,
                                            digitiser=test_digitiser)

    cross_types = json.load(open(f"./train-configs/specs/cross.json", 'r'))[spec_type]
    noiserange = [0.0, 1.0]
    with open(f"./train-configs/specs/cross-our.py", "r") as f:
        code = f.read()
    exec(code)


