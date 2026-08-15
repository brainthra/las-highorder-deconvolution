import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
import gc

# region Imports
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


# target_comps = "gauss_gauss" # ["gauss_gauss", "boltz_boltz", "boltz_gauss"]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--target_comps", type=str, required=True, choices=["gauss_gauss", "boltz_boltz", "boltz_gauss"])

    return parser.parse_args()


args = parse_args()

Bmin = 10
Bmax = 65535
Bstep = 1
Btype = np.int64

Emin = 0.001
Emax = 1.0
target_bins = 100
ext_factor = 5.0

pred_bins_target = 20
pred_tar_scale = "lin"
pred_bins_ext = 10
pred_ext_scale = "log"

seed = args.seed

target_comps = args.target_comps

reconstructor = pickle.load(open(f"./training/{Bmin}-{Bmax}-{Bstep}/{Emin}-{Emax}--{target_bins}--{ext_factor}/existing/shighorder/noise--0.0-1.0/flux--0.01-1.0/mlp--20-lin--10-log--10x/{seed}/model/our.pkl", 'rb'))

np.random.seed(seed)
torch.random.manual_seed(seed)

# energies
target_range = (Emin, Emax)

TARGET_ENERGIES = np.linspace(*target_range, target_bins)
FULL_ENERGIES = np.linspace(target_range[1], target_range[1]*ext_factor, int(target_bins*(ext_factor-1.0))+1)
FULL_ENERGIES = np.concatenate((TARGET_ENERGIES, FULL_ENERGIES[1:]))

# design
design:Spectrometer

with open(f"./train-configs/designs/existing.py", 'r') as f:
    code = f.read()

exec(code)

# spectra generation
spec_type = 'shighorder'
configs = {}
sgm = 0.2

# analytical params
T_min, T_max = configs.get("T", {}).get("min", 0.1), configs.get("T", {}).get("max", 0.8)
mu_min, mu_max = configs.get("mu", {}).get("min", 0.150), configs.get("mu", {}).get("max", 0.750)
sg_min, sg_max = configs.get("sigma", {}).get("min", sgm), configs.get("sigma", {}).get("max", sgm)
alpha_min, alpha_max = configs.get("alpha", {}).get("min", 0), configs.get("alpha", {}).get("max", 0)

# weighting
bamp_min, bamp_max = configs.get("boltz_amp", {}).get("min", 0.3), configs.get("boltz_amp", {}).get("max", 1.0)
gamp_min, gamp_max = configs.get("gauss_amp", {}).get("min", 0.3), configs.get("gauss_amp", {}).get("max", 1.0)

# sampling limits
n_gauss = configs.get("n_gauss", 2 if target_comps == "gauss_gauss" else 1 if target_comps == "boltz_gauss" else 0)
n_boltz = configs.get("n_boltz", 2 if target_comps == "boltz_boltz" else 1 if target_comps == "boltz_gauss" else 0)
max_comps = configs.get("max_comps", 10)

# number of samples
samples = configs.get("samples", 75_000)


temp_min, temp_max = T_min * Emax, T_max * Emax
mu_min, mu_max = mu_min * Emax, mu_max * Emax
sg_min, sg_max = sg_min * Emax, sg_max * Emax

print(f"T in [{temp_min:.3f}, {temp_max:.3f}]")
print(f"mu in [{mu_min:.3f}, {mu_max:.3f}]")
print(f"sg in [{sg_min:.3f}, {sg_max:.3f}]")
print(f"alpha in [{alpha_min:.3f}, {alpha_max:.3f}]")
print(f"boltz_amp in [{bamp_min:.3f}, {bamp_max:.3f}]")
print(f"gauss_amp in [{gamp_min:.3f}, {gamp_max:.3f}]") 
print(f"max n_boltz: {n_boltz}, max n_gauss: {n_gauss}, max_comps: {max_comps}")

np.random.seed(seed)
spectra_set, spectra_params = CompositeGenerator(TARGET_ENERGIES, FULL_ENERGIES, temp_lim=[temp_min, temp_max], mu_lim=(mu_min, mu_max), sg_lim=(sg_min, sg_max), alpha_lim=[alpha_min, alpha_max], boltz_amp_lim=(bamp_min, bamp_max), gauss_amp_lim=(gamp_min, gamp_max)).generate(samples, max_boltz=n_boltz, max_gauss=n_gauss, max_comps=max_comps, random_state=seed)

param_str = f"{spec_type}--T-{temp_min:.3f}-{temp_max:.3f}--mu-{mu_min:.3f}-{mu_max:.3f}--sg-{sg_min:.3f}-{sg_max:.3f}--al-{alpha_min:.3f}-{alpha_max:.3f}--bamp-{bamp_min:.3f}-{bamp_max:.3f}--gamp-{gamp_min:.3f}-{gamp_max:.3f}"

codestr = 'title_str = f"Test # {idx} | $\\\phi$: {fluxes[idx]:.2e}\\n" + f"n_gauss: {spectra_params[idx][\'n_gauss\']}, n_boltz: {spectra_params[idx][\'n_boltz\']}"'

target_idxs = np.linspace(0, target_bins-1, pred_bins_target).astype(int).tolist() if pred_tar_scale == 'lin' else np.geomspace(0.1, target_bins-1, pred_bins_target).astype(int).tolist()
extended_idxs = np.geomspace(target_bins-1, int(target_bins*ext_factor)-1, pred_bins_ext+1).astype(int).tolist() if pred_ext_scale == 'log' else np.linspace(target_bins-1, int(target_bins*ext_factor)-1, pred_bins_ext+1).astype(int).tolist()


sample_idxs = target_idxs + extended_idxs
sample_idxs = np.array(sorted(list(set(sample_idxs))))

two_comps_indices = []
two_comps_params = []

c = 0

for i in range(len(spectra_params)):
    if target_comps == "boltz_boltz" and spectra_params[i]["n_gauss"] == 0 and spectra_params[i]["n_boltz"] == 2:
        two_comps_params.append(spectra_params[i])
        two_comps_indices.append(i)
    elif target_comps == "gauss_gauss" and spectra_params[i]["n_gauss"] == 2 and spectra_params[i]["n_boltz"] == 0:
        two_comps_params.append(spectra_params[i])
        two_comps_indices.append(i)
    elif target_comps == "boltz_gauss" and spectra_params[i]["n_gauss"] == 1 and spectra_params[i]["n_boltz"] == 1:
        two_comps_params.append(spectra_params[i])
        two_comps_indices.append(i)
    else:
        c += 1

two_comps_params = np.array(two_comps_params)
two_comps_indices = np.array(two_comps_indices)

spec_params, spec_indices = two_comps_params[:15_000], two_comps_indices[:15_000]

print(spectra_set[spec_indices[-1]])

ds = LASDataset(spectra_set[spec_indices], design, sample_idxs=sample_idxs)

digitiser = Camera(Bmin, Bmax, Bstep, Btype)
optimal_flux = int(ds.get_optimal_flux(digitiser, up_offset=0.05))
max_flux = optimal_flux * 1.0
min_flux = optimal_flux * 0.01

MUS = np.linspace(0.01, 1.0, 100)

recon_errs = []

np.random.seed(seed)
mutator = Mutator(digitiser, [min_flux, max_flux], [0.0, 0.0], "linear", "linear", 0.1)
modded_ms, fluxes = mutator.prepare(ds.measureds, [min_flux, max_flux], (0.0, 0.0), return_flux=True)

for i in range(len(spec_indices)):
    p = spec_params[i]
    spectra_in = spectra_set[spec_indices[i]]
    spectra_in = spectra_in * fluxes[i]  
    
    M = modded_ms[i]
    
    pred = reconstructor.inference(M)
    spectra_in = np.array(spectra_in[:100]).reshape(1, -1)
    mea = np.mean(np.abs(spectra_in - pred)) / fluxes[i]
    recon_errs.append(mea)


recon_errs_nse = []

np.random.seed(seed)
mutator = Mutator(digitiser, [min_flux, max_flux], [0.0, 1.0], "linear", "linear", 0.1)
modded_ms, fluxes = mutator.prepare(ds.measureds, [min_flux, max_flux], (0.0, 1.0), return_flux=True)

for i in range(len(spec_indices)):
    p = spec_params[i]
    spectra_in = spectra_set[spec_indices[i]]
    spectra_in = spectra_in * fluxes[i]  
    
    M = modded_ms[i]

    pred = reconstructor.inference(M)
    spectra_in = np.array(spectra_in[:100]).reshape(1, -1)
    mea = np.mean(np.abs(spectra_in - pred)) / fluxes[i]
    recon_errs_nse.append(mea)

os.makedirs(f"./bench-our/{seed}", exist_ok=True)

np.save(
    f"./bench-our/{seed}/{target_comps}-our-clean.npy",
    recon_errs
)

np.save(
    f"./bench-our/{seed}/{target_comps}-our-noised.npy",
    recon_errs_nse
)