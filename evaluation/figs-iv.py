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

from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, LogLocator, NullFormatter

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


lim = "our"

Bmin, Bmax, Bstep, Btype = 10, 65535, 1, np.int64
Emin, Emax, target_bins, ext_factor = 0.001, 1.0, 100, 5.0
design, spec_type = "existing", "shighorder"
noise_min, noise_max = 0.0, 1.0
flux_min, flux_max = 0.01, 1.0
model_type, pred_bins_target, pred_tar_scale, pred_bins_ext, pred_ext_scale, ensembles = "mlp", 20, "lin", 10, "log", 10
seed = 42

target_range = (Emin, Emax)


digitiser = Camera(10, 65535, 1, np.int64)

TARGET_ENERGIES = np.linspace(*target_range, target_bins)
FULL_ENERGIES = np.linspace(target_range[1], target_range[1]*ext_factor, int(target_bins*(ext_factor-1.0))+1)
FULL_ENERGIES = np.concatenate((TARGET_ENERGIES, FULL_ENERGIES[1:]))

np.random.seed(seed)
torch.random.manual_seed(seed)


def add_strip(ax, color="C0", height=0.1, y=1.12):
    # colored strip
    ax.add_patch(
        Rectangle(
            (0, y), 
            1, height,       
            transform=ax.transAxes,
            clip_on=False,
            facecolor=color,
            edgecolor="none",
            zorder=10,
        )
    )

def fig_to_np(fig, *, rgb=True):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    w, h = canvas.get_width_height()
    arr = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    arr = arr.reshape(h, w, 4)

    return arr[:, :, :3].copy() if rgb else arr.copy()

def np_to_fig(arr, *, figsize=None, dpi=100):
    arr = np.asarray(arr)

    if figsize is None:
        h, w = arr.shape[:2]
        figsize = (w / dpi, h / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(arr)
    ax.axis("off")
    fig.tight_layout(pad=0)

    return fig


# region load designs

design:Spectrometer

with open(f"../train-configs/designs/idealised.py", 'r') as f:
    code = f.read()

exec(code)

design_idl = design

design:Spectrometer

with open(f"../train-configs/designs/existing.py", 'r') as f:
    code = f.read()

exec(code)

# endregion

# region load spectra
spectra_set:np.ndarray
spectra_params: dict

configs = json.load(open(f"../train-configs/specs/{spec_type}/{lim}.json", 'r'))
with open(f"../train-configs/specs/{spec_type}/init.py", "r") as f:
    code = f.read()

exec(code)
# endregion

# region dataset preparation and reconstructor loading
target_idxs = np.linspace(0, target_bins-1, pred_bins_target).astype(int).tolist() if pred_tar_scale == 'lin' else np.geomspace(0.1, target_bins-1, pred_bins_target).astype(int).tolist()
extended_idxs = np.geomspace(target_bins-1, int(target_bins*ext_factor)-1, pred_bins_ext+1).astype(int).tolist() if pred_ext_scale == 'log' else np.linspace(target_bins-1, int(target_bins*ext_factor)-1, pred_bins_ext+1).astype(int).tolist()

sample_idxs = target_idxs + extended_idxs
sample_idxs = np.array(sorted(list(set(sample_idxs))))

ds = LASDataset(spectra_set, design, sample_idxs=sample_idxs)
digitiser = Camera(Bmin, Bmax, Bstep, Btype)
optimal_flux = int(ds.get_optimal_flux(digitiser, up_offset=0.05))
max_flux = optimal_flux * flux_max
min_flux = optimal_flux * flux_min
reconstructor:MultiReconstructor = pickle.load(open(f'../training/10-65535-1/0.001-1.0--100--5.0/existing/{spec_type}/noise--0.0-1.0/flux--0.01-1.0/mlp--20-lin--10-log--10x/{seed}/model/{lim}.pkl', 'rb'))
train_ds, val_ds, test_ds = ds.split([0.6, 0.15, 0.25], shuffle=False)
train_ds.prepare(reconstructor.out_scaler)
val_ds.prepare(reconstructor.out_scaler)
test_ds.prepare(reconstructor.out_scaler)

ds_idl = LASDataset(spectra_set, design_idl, sample_idxs=sample_idxs)
optimal_flux_idl = int(ds_idl.get_optimal_flux(digitiser, up_offset=0.05))
max_flux_idl = optimal_flux_idl * flux_max
min_flux_idl = optimal_flux_idl * flux_min
reconstructor_idl:MultiReconstructor = pickle.load(open(f'../training/10-65535-1/0.001-1.0--100--5.0/idealised/{spec_type}/noise--0.0-1.0/flux--0.01-1.0/mlp--20-lin--10-log--10x/{seed}/model/{lim}.pkl', 'rb'))
train_ds_idl, val_ds_idl, test_ds_idl = ds_idl.split([0.6, 0.15, 0.25], shuffle=False)
train_ds_idl.prepare(reconstructor_idl.out_scaler)
val_ds_idl.prepare(reconstructor_idl.out_scaler)
test_ds_idl.prepare(reconstructor_idl.out_scaler)

optimal_flux /100

# endregion

# region fig 7

clrs = ["C0", "C1",  "C4", "C5", "C6", "C8", "C9"]

recon_maes = np.load(
    f'../testing/10-65535-1/0.001-1.0--100--5.0/existing/shighorder/'
    f'noise--0.0-1.0/flux--0.01-1.0/mlp--20-lin--10-log--10x/{seed}/res/'
    f'mae--B-{digitiser.visible}-{digitiser.saturation}-{digitiser.step}'
    f'--flux-0.1-0.1--noise-0.0-1.0--{lim}.npy'
)

recon_mrp = np.load(
    f'../testing/10-65535-1/0.001-1.0--100--5.0/existing/shighorder/'
    f'noise--0.0-1.0/flux--0.01-1.0/mlp--20-lin--10-log--10x/{seed}/res/'
    f'mrp--B-{digitiser.visible}-{digitiser.saturation}-{digitiser.step}'
    f'--flux-0.1-0.1--noise-0.0-1.0--{lim}.npy'
)

min_flux, max_flux = optimal_flux * 0.1, optimal_flux * 0.1
min_noise, max_noise = 0.0, 1.0

mutator = Mutator(
    digitiser,
    (min_flux, max_flux),
    (min_noise, max_noise),
    "linear",
    "linear"
)

np.random.seed(seed)

modded_ms, fluxes = mutator.prepare(
    test_ds.measureds,
    (min_flux, max_flux),
    (min_noise, max_noise),
    return_flux=True
)

modded_ss = test_ds.spectra * fluxes[:, np.newaxis]


idcs = [ 109150, 4982, 33539, 93821, 769, 29747]

print(f"Selected indices for plotting: {idcs}")

np.random.seed(42)

rows = 3
cols = 6

erous = []

fig, ax = plt.subplots(
    rows,
    cols,
    figsize=(15, 5),
    gridspec_kw={'height_ratios': [2, 1, 1]},
    sharey='row'
)

fig.set_dpi(200)


for i, idx in enumerate(idcs):

    pred, err, info = reconstructor.inference(
        modded_ms[idx],
        method="opt",
        digitiser=digitiser,
        return_err=True,
        return_info=True,
        debug=False
    )

    # ==========================================================
    # Row 1: Spectrum
    # ==========================================================

    ax[0, i].plot(
        TARGET_ENERGIES,
        modded_ss[idx][:target_bins],
        label="$S$"
    )

    ax[0, i].plot(
        TARGET_ENERGIES,
        pred.squeeze(),
        "--",
        label="$\\hat{S}$",
        c="orange"
    )

    ax[0, i].fill_between(
        TARGET_ENERGIES,
        pred.squeeze() - err.squeeze(),
        pred.squeeze() + err.squeeze(),
        alpha=0.5,
        color="orange"
    )

    ax[0, i].ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0)
    )

    ax[0, i].set_ylim(0, 8e3)
    ax[0, i].set_xlabel("Energy (MeV)")

    # ==========================================================
    # Row 2: Measurement
    # ==========================================================

    measurement_pred = reconstructor.spectrometer(info.spec_full)

    ax[1, i].scatter(
        range(1, len(modded_ms[idx]) + 1),
        modded_ms[idx],
        label="$M$"
    )

    ax[1, i].scatter(
        range(1, len(modded_ms[idx]) + 1),
        measurement_pred,
        color="orange",
        marker="x",
        label="$\\hat{M}$"
    )

    ax[1, i].set_yscale("log")

    ax[1, i].set_xticks(
        range(1, len(modded_ms[idx]) + 1)
    )

    
    # ==========================================================
    # Row 3: Relative measurement error
    # ==========================================================

    measurement_error = (
        100
        * (measurement_pred.astype(int) - modded_ms[idx])
        / modded_ms[idx]
    )

    ax[2, i].stem(
        range(1, len(modded_ms[idx]) + 1),
        measurement_error,
        markerfmt="x",
        linefmt="C1-",
        basefmt="C0-"
    )

    ax[2, i].set_xticks(
        range(1, len(modded_ms[idx]) + 1)
    )

    ax[2, i].set_ylim(-5, 5)
    ax[2, i].set_xlabel("Scintillator #")

    if i == 0:
        ax[0, i].set_ylabel("Spectral counts")
        ax[1, i].set_ylabel("counts")
        ax[2, i].set_ylabel(
            "                              $\\bf{Measurement}$ \n"
            "Relative error (%)         "
        )

        ax[0, i].legend(fontsize=12)
        ax[1, i].legend(fontsize=7)

    mea = (
        mae(
            modded_ss[idx][:target_bins].reshape(1, -1),
            pred.reshape(1, -1)
        )
        / fluxes[idx]
    )

    erous.append(mea)


for a in ax[1, :]:

    a.yaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0,))
    )

    a.yaxis.set_major_formatter(
        LogFormatterMathtext(base=10.0, labelOnlyBase=True)
    )

    a.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1
        )
    )

    a.yaxis.set_minor_formatter(
        NullFormatter()
    )

for i in range(cols):
        add_strip(ax[0, i], color=f"{clrs[i]}")

fig.tight_layout()


img1 = fig_to_np(fig)


plt.figure(figsize=(3, 5), dpi=200)

vio_col = "gray"
vds = []

for seed in [42]:

    vd = np.load(
        f'../testing/10-65535-1/0.001-1.0--100--5.0/existing/shighorder/'
        f'noise--0.0-1.0/flux--0.01-1.0/mlp--20-lin--10-log--10x/'
        f'{seed}/res/'
        f'mae--B-10-65535-1--flux-0.1-0.1--noise-0.0-1.0--our.npy'
    )

    vds.append(vd)


vd_tot = np.array(np.concatenate(vds))

print(vd_tot.shape)


c = plt.violinplot(
    np.log10(vd_tot)
)

c['cbars'].set_edgecolor(vio_col)
c['cmins'].set_edgecolor(vio_col)
c['cmaxes'].set_edgecolor(vio_col)

for pc in c['bodies']:
    pc.set_facecolor(vio_col)
    pc.set_edgecolor(vio_col)
    pc.set_alpha(0.3)


plt.scatter(
    1,
    np.log10(np.mean(vd_tot)),
    color=vio_col,
    s=100
)


plt.scatter(
    [[1], [1], [1], [1], [1], [1]],
    np.log10(vd_tot[idcs]),
    c=clrs[:6],
    s=100,
    marker='X',
    zorder=10
)


plt.suptitle(
    f"          Test $\\phi = \\phi^* / 10$",
    fontsize=18
)

plt.ylabel(
    "log10 ( $\\epsilon_S$ )",
    fontsize=14
)

plt.xticks(
    [1],
    ["existing"],
    fontsize=12
)

plt.yticks(fontsize=12)

plt.xlabel(
    "Design",
    fontsize=14
)


plt.axhline(
    np.log10(1e-3),
    color='red',
    linestyle='--',
    alpha=0.5
)

plt.axhline(
    np.log10(1e-4),
    color='green',
    linestyle='--',
    alpha=0.5
)


plt.tight_layout()


plt.text(
    1.1,
    np.log10(1e-3) + 0.1,
    f"{100 * len(np.where(vd_tot > 1e-3)[0]) / len(vd_tot):.1f}%",
    fontsize=12,
    va='center'
)

plt.text(
    1.1,
    -3.5,
    f"{100 * len(np.where((vd_tot > 1e-4) & (vd_tot < 1e-3))[0]) / len(vd_tot):.1f}%",
    fontsize=12,
    va='center'
)

plt.text(
    1.1,
    np.log10(1e-4) - 0.1,
    f"{100 * len(np.where(vd_tot < 1e-4)[0]) / len(vd_tot):.1f}%",
    fontsize=12,
    va='center'
)


img2 = fig_to_np(plt.gcf())


img_combined = np.hstack(
    (img2, img1)
)

fig = np_to_fig(img_combined)

fig.text(
    0.005,
    0.98,
    "(a)",
    fontsize=40,
    # fontweight='bold',
    va='top',
    ha='left'
)

fig.text(
    0.17,
    0.98,
    "(b)",
    fontsize=40,
    # fontweight='bold',
    va='top',
    ha='left'
)


fig.savefig(
    "fig-7.pdf",
    dpi=300,
    bbox_inches='tight'
)

# endregion

# region fig 8
digitiser = Camera(10, 65535, 1, np.int64)

recon_maes = np.load(
    f'../testing/10-65535-1/0.001-1.0--100--5.0/existing/shighorder/'
    f'noise--0.0-1.0/flux--0.01-1.0/mlp--20-lin--10-log--10x/{seed}/res/'
    f'mae--B-{digitiser.visible}-{digitiser.saturation}-{digitiser.step}'
    f'--flux-0.1-0.1--noise-0.0-1.0--{lim}.npy'
)

recon_mrp = np.load(
    f'../testing/10-65535-1/0.001-1.0--100--5.0/existing/shighorder/'
    f'noise--0.0-1.0/flux--0.01-1.0/mlp--20-lin--10-log--10x/{seed}/res/'
    f'mrp--B-{digitiser.visible}-{digitiser.saturation}-{digitiser.step}'
    f'--flux-0.1-0.1--noise-0.0-1.0--{lim}.npy'
)

min_flux, max_flux = optimal_flux * 0.1, optimal_flux * 0.1
min_noise, max_noise = 0.0, 1.0

mutator = Mutator(
    digitiser,
    (min_flux, max_flux),
    (min_noise, max_noise),
    "linear",
    "linear"
)

mutator_idl = Mutator(
    digitiser,
    (min_flux_idl, max_flux_idl),
    (min_noise, max_noise),
    "linear",
    "linear"
)

np.random.seed(42)

modded_ms, fluxes = mutator.prepare(
    test_ds.measureds,
    (min_flux, max_flux),
    (min_noise, max_noise),
    return_flux=True
)

modded_ss = test_ds.spectra * fluxes[:, np.newaxis]

np.random.seed(42)

modded_ms_idl, fluxes_idl = mutator_idl.prepare(
    test_ds_idl.measureds,
    (optimal_flux_idl * 0.1, optimal_flux_idl * 0.1),
    (min_noise, max_noise),
    return_flux=True
)

modded_ss_idl = test_ds_idl.spectra * fluxes_idl[:, np.newaxis]


idcs = [ 109150, 4982, 33539, 93821, 769, 29747]

np.random.seed(42)

rows = 2
cols = 6

erous = []

fig, ax = plt.subplots(
    rows,
    cols,
    figsize=(15, 5),
    sharey='row'
)

fig.set_dpi(200)


for i, idx in enumerate(idcs):

    pred, err, info = reconstructor.inference(
        modded_ms[idx],
        method="opt",
        digitiser=digitiser,
        return_err=True,
        return_info=True,
        debug=False
    )

    pred_idl, err_idl, info_idl = reconstructor_idl.inference(
        modded_ms_idl[idx],
        method="opt",
        digitiser=digitiser,
        return_err=True,
        return_info=True,
        debug=False
    )

    # ==========================================================
    # Row 1: Existing design spectrum
    # ==========================================================

    ax[0, i].plot(
        TARGET_ENERGIES,
        modded_ss[idx][:target_bins],
        label="$S$"
    )

    ax[0, i].plot(
        TARGET_ENERGIES,
        pred.squeeze(),
        "--",
        label="$\\hat{S}$",
        c="orange"
    )

    ax[0, i].fill_between(
        TARGET_ENERGIES,
        pred.squeeze() - err.squeeze(),
        pred.squeeze() + err.squeeze(),
        alpha=0.5,
        color="orange"
    )

    ax[0, i].ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0)
    )

    ax[0, i].set_xticklabels([])

    # ==========================================================
    # Row 2: Idealised design spectrum
    # ==========================================================
    
    ax[1, i].plot(
        TARGET_ENERGIES,
        modded_ss_idl[idx][:target_bins],
        label="$S$"
    )
    
    ax[1, i].plot(
        TARGET_ENERGIES,
        pred_idl.squeeze(),
        "--",
        label="$\\hat{S}$",
        c="orange"
    )
    
    ax[1, i].fill_between(
        TARGET_ENERGIES,
        pred_idl.squeeze() - err_idl.squeeze(),
        pred_idl.squeeze() + err_idl.squeeze(),
        alpha=0.5,
        color="orange"
    )
    
    ax[1, i].ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0)
    )
    
    # ax[0, i].set_ylim(0, 8e3)
    ax[1, i].set_xlabel("Energy (MeV)")


fig.tight_layout()


img1 = fig_to_np(fig)


plt.figure(figsize=(3, 5), dpi=200)

vio_col = "gray"
vds = []

for seed in [42]:

    vd = np.load(
        f'../testing/10-65535-1/0.001-1.0--100--5.0/existing/shighorder/'
        f'noise--0.0-1.0/flux--0.01-1.0/mlp--20-lin--10-log--10x/'
        f'{seed}/res/'
        f'mae--B-10-65535-1--flux-0.1-0.1--noise-0.0-1.0--our.npy'
    )

    vds.append(vd)
vd_tot = np.array(np.concatenate(vds))
print(vd_tot.shape)
c = plt.violinplot(np.log10(vd_tot), positions=[1], widths=0.75)
c['cbars'].set_edgecolor(vio_col)
c['cmins'].set_edgecolor(vio_col)
c['cmaxes'].set_edgecolor(vio_col)
for pc in c['bodies']:
    pc.set_facecolor(vio_col)
    pc.set_edgecolor(vio_col)
    pc.set_alpha(0.3)
plt.scatter(1, np.log10(np.mean(vd_tot)), color=vio_col, s=100)

vds = []
for seed in [42]:

    vd = np.load(
        f'../testing/10-65535-1/0.001-1.0--100--5.0/idealised/shighorder/'
        f'noise--0.0-1.0/flux--0.01-1.0/mlp--20-lin--10-log--10x/'
        f'{seed}/res/'
        f'mae--B-10-65535-1--flux-0.1-0.1--noise-0.0-1.0--our.npy'
    )

    vds.append(vd)
vd_tot = np.array(np.concatenate(vds))
print(vd_tot.shape)
c = plt.violinplot(np.log10(vd_tot), positions=[2], widths=0.75)
c['cbars'].set_edgecolor(vio_col)
c['cmins'].set_edgecolor(vio_col)
c['cmaxes'].set_edgecolor(vio_col)
for pc in c['bodies']:
    pc.set_facecolor(vio_col)
    pc.set_edgecolor(vio_col)
    pc.set_alpha(0.3)
plt.scatter(2, np.log10(np.mean(vd_tot)), color=vio_col, s=100)


plt.xticks([1, 2], ["Existing", "     Idealised"], fontsize=14)
plt.suptitle(f"          Test $\phi = \phi^* / 10$", fontsize=18)
plt.ylabel("log10 ( $\epsilon_S$ )", fontsize=16)
plt.xlabel("Design", fontsize=18)
plt.yticks(fontsize=14)

plt.axhline(np.log10(1e-3), color='red', linestyle='--', alpha=0.5)
plt.axhline(np.log10(1e-4), color='green', linestyle='--', alpha=0.5)
plt.xlim(0.5, 2.5)
plt.tight_layout()

img2 = fig_to_np(plt.gcf())

gap = 36  # pixels

img_combined = np.hstack(
    (img2, np.ones((img2.shape[0], gap, 3), dtype=np.uint8) * 255, img1)
)


fig = np_to_fig(img_combined)

fig.text(
    0.005,
    1.02,
    "(a)",
    fontsize=40,
    # fontweight='bold',
    va='top',
    ha='left'
)

fig.text(
    0.163,
    1.02,
    "(b)",
    fontsize=40,
    # fontweight='bold',
    va='top',
    ha='left'
)

fig.savefig(
    "fig-10.pdf",
    dpi=300,
    bbox_inches='tight'
)

# endregion


