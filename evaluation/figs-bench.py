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
import pandas as pd

from matplotlib.patches import Rectangle, Circle
from matplotlib.legend_handler import HandlerPatch

lim = "our"

Bmin, Bmax, Bstep, Btype = 10, 65535, 1, 'int'
Emin, Emax, target_bins, ext_factor = 0.001, 1.0, 100, 5.0
design, spec_type = 'existing', 'shighorder'
noise_min, noise_max = 0.0, 1.0
flux_min, flux_max = 0.01, 1.0
model_type, pred_bins_target, pred_tar_scale, pred_bins_ext, pred_ext_scale, ensembles = 'mlp', 20, 'lin', 10, 'log', 10

def colored_violinplot(ax, data, positions, colors, widths=0.8, border_colors=None):
    parts = ax.violinplot(data, positions=positions, widths=widths)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor(border_colors[i] if border_colors else colors[i])
        pc.set_alpha(0.5)
        pc
    parts["cbars"].set_edgecolor(border_colors[i] if border_colors else colors[i])    
    parts["cmins"].set_edgecolor(border_colors[i] if border_colors else colors[i])
    parts["cmaxes"].set_edgecolor(border_colors[i] if border_colors else colors[i])
    return parts


class HandlerRectWithCircle(HandlerPatch):
    def create_artists(
        self, legend, orig_handle,
        xdescent, ydescent, width, height, fontsize, trans
    ):
        rect = Rectangle(
            (xdescent, ydescent),
            width,
            height,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            transform=trans
        )

        circle = Circle(
            (xdescent + width / 2, ydescent + height / 2),
            radius=min(width, height) * 0.25,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            transform=trans
        )

        return [rect, circle]

def print_comparison_table(df):
    target_labels = {
        "boltz_boltz": "Boltz.-Boltz.",
        "boltz_gauss": "Boltz.-Gauss.",
        "gauss_gauss": "Gauss.-Gauss.",
    }

    noise_labels = {
        "clean": "0",
        "noised": "[-1,1]",
    }

    target_order = [
        "boltz_boltz",
        "boltz_gauss",
        "gauss_gauss",
    ]

    noise_order = [
        "clean",
        "noised",
    ]

    model_order = [
        "our",
        "mpd",
    ]

    print("\nComparison of reconstruction performance\n")
    print("Errors: x10^-3\n")

    header = (
        f"{'Set':<18}"
        f"{'Noise':<10}"
        f"{'Method':<18}"
        f"{'Mean':>10}"
        f"{'Median':>10}"
        f"{'IQR':>10}"
        f"{'Poor (%)':>12}"
        f"{'Accur. U Accept. (%)':>22}"
    )

    print(header)
    print("-" * len(header))

    for target in target_order:
        dft = df[df["Target"] == target]

        first_target_row = True

        for noise in noise_order:
            dfn = dft[dft["Noise"] == noise]

            first_noise_row = True

            for model in model_order:
                row = dfn[dfn["model"] == model]

                if row.empty:
                    continue

                row = row.iloc[0]

                target_label = (
                    target_labels.get(target, target)
                    if first_target_row
                    else ""
                )

                noise_label = (
                    noise_labels.get(noise, noise)
                    if first_noise_row
                    else ""
                )

                # Change this if your model names differ
                model_label = (
                    "Proposed"
                    if model == "our"
                    else "MPD"
                )

                print(
                    f"{target_label:<18}"
                    f"{noise_label:<10}"
                    f"{model_label:<18}"
                    f"{row['mean'] * 1e3:>10.3f}"
                    f"{row['median'] * 1e3:>10.3f}"
                    f"{row['IQR'] * 1e3:>10.3f}"
                    f"{row['fail_rate']:>12.4f}"
                    f"{row['non_fail_rate']:>22.4f}"
                )

                first_target_row = False
                first_noise_row = False

            print("-" * len(header))

        print("=" * len(header))


df = pd.DataFrame(columns=["Target", "Noise", "model", "mean", "std", "min", "max", "median", "IQR", "fail_rate", "non_fail_rate", "accept_rate", "accu_rate"])

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for i in range(3):
    ax = axs[i]
    mdl_errs = []
    cda_errs = []
    mdl_errs_nse = []
    cda_errs_nse = []
    if i == 0:
        for seed in [0, 1, 10, 33, 42]:
            mdl_errs.append(np.load(f"../bench-our/{seed}/boltz_boltz-our-clean.npy"))
            cda_errs.append(np.load(f"../bench-mpd/{seed}/boltz_boltz-mpd-clean.npy"))
            mdl_errs_nse.append(np.load(f"../bench-our/{seed}/boltz_boltz-our-noised.npy"))
            cda_errs_nse.append(np.load(f"../bench-mpd/{seed}/boltz_boltz-mpd-noised.npy"))
        ttl = "      Boltzmann + Boltzmann"
        tget = "boltz_boltz"
        ax.set_ylabel("Log10($\epsilon_{S}$)", fontsize=14)
    elif i == 1:
        for seed in [0, 1, 10, 33, 42]:
            mdl_errs.append(np.load(f"../bench-our/{seed}/boltz_gauss-our-clean.npy"))
            cda_errs.append(np.load(f"../bench-mpd/{seed}/boltz_gauss-mpd-clean.npy"))
            mdl_errs_nse.append(np.load(f"../bench-our/{seed}/boltz_gauss-our-noised.npy"))
            cda_errs_nse.append(np.load(f"../bench-mpd/{seed}/boltz_gauss-mpd-noised.npy"))
        ttl = "    Boltzmann + Gaussian"
        tget = "boltz_gauss"
    else:
        for seed in [0, 1, 10, 33, 42]:
            mdl_errs.append(np.load(f"../bench-our/{seed}/gauss_gauss-our-clean.npy"))
            cda_errs.append(np.load(f"../bench-mpd/{seed}/gauss_gauss-mpd-clean.npy"))
            mdl_errs_nse.append(np.load(f"../bench-our/{seed}/gauss_gauss-our-noised.npy"))
            cda_errs_nse.append(np.load(f"../bench-mpd/{seed}/gauss_gauss-mpd-noised.npy"))
        ttl = "    Gaussian + Gaussian"
        tget = "gauss_gauss"
    
    mdl_errs = np.concatenate(mdl_errs)
    cda_errs = np.concatenate(cda_errs)
    mdl_errs_nse = np.concatenate(mdl_errs_nse)
    cda_errs_nse = np.concatenate(cda_errs_nse)

    df.loc[len(df)] = {
        "Target": tget,
        "Noise": "clean",
        "model": "our",
        "mean": np.mean(mdl_errs),
        "std": np.std(mdl_errs),
        "min": np.min(mdl_errs),
        "max": np.max(mdl_errs),
        "median": np.median(mdl_errs),
        "IQR": np.percentile(mdl_errs, 75) - np.percentile(mdl_errs, 25),
        "fail_rate": len(np.where((mdl_errs > 1e-3))[0]) / len(mdl_errs),
        "non_fail_rate": len(np.where((mdl_errs <= 1e-3))[0]) / len(mdl_errs),
        "accept_rate": len(np.where((mdl_errs <= 1e-3) & (mdl_errs > 1e-4))[0]) / len(mdl_errs),
        "accu_rate": len(np.where((mdl_errs <= 1e-4))[0]) / len(mdl_errs)
    }

    df.loc[len(df)] = {
        "Target": tget,
        "Noise": "clean",
        "model": "mpd",
        "mean": np.mean(cda_errs),
        "std": np.std(cda_errs),
        "min": np.min(cda_errs),
        "max": np.max(cda_errs),
        "median": np.median(cda_errs),
        "IQR": np.percentile(cda_errs, 75) - np.percentile(cda_errs, 25),
        "fail_rate": len(np.where((cda_errs > 1e-3))[0]) / len(cda_errs),
        "non_fail_rate": len(np.where((cda_errs <= 1e-3))[0]) / len(cda_errs),
        "accept_rate": len(np.where((cda_errs <= 1e-3) & (cda_errs > 1e-4))[0]) / len(cda_errs),
        "accu_rate": len(np.where((cda_errs <= 1e-4))[0]) / len(cda_errs)
    }

    df.loc[len(df)] = {
        "Target": tget,
        "Noise": "noised",
        "model": "our",
        "mean": np.mean(mdl_errs_nse),
        "std": np.std(mdl_errs_nse),
        "min": np.min(mdl_errs_nse),
        "max": np.max(mdl_errs_nse),
        "median": np.median(mdl_errs_nse),
        "IQR": np.percentile(mdl_errs_nse, 75) - np.percentile(mdl_errs_nse, 25),
        "fail_rate": len(np.where((mdl_errs_nse > 1e-3))[0]) / len(mdl_errs_nse),
        "non_fail_rate": len(np.where((mdl_errs_nse <= 1e-3))[0]) / len(mdl_errs_nse),
        "accept_rate": len(np.where((mdl_errs_nse <= 1e-3) & (mdl_errs_nse > 1e-4))[0]) / len(mdl_errs_nse),
        "accu_rate": len(np.where((mdl_errs_nse <= 1e-4))[0]) / len(mdl_errs_nse)
    }

    df.loc[len(df)] = {
        "Target": tget,
        "Noise": "noised",
        "model": "mpd",
        "mean": np.mean(cda_errs_nse),
        "std": np.std(cda_errs_nse),
        "min": np.min(cda_errs_nse),
        "max": np.max(cda_errs_nse),
        "median": np.median(cda_errs_nse),
        "IQR": np.percentile(cda_errs_nse, 75) - np.percentile(cda_errs_nse, 25),
        "fail_rate": len(np.where((cda_errs_nse > 1e-3))[0]) / len(cda_errs_nse),
        "non_fail_rate": len(np.where((cda_errs_nse <= 1e-3))[0]) / len(cda_errs_nse),
        "accept_rate": len(np.where((cda_errs_nse <= 1e-3) & (cda_errs_nse > 1e-4))[0]) / len(cda_errs_nse),
        "accu_rate": len(np.where((cda_errs_nse <= 1e-4))[0]) / len(cda_errs_nse)
    }


    ax.set_title(ttl, fontsize=16)
    colored_violinplot(ax, np.log10(mdl_errs), positions=[0], colors=['none'], widths=0.8, border_colors=['k'])
    ax.scatter([0], [np.log10(np.mean(mdl_errs))], color='k', label="Mean Model Error")
    ax.text(0-1.5, np.log10(1e-3)+0.6, f"{100*len(np.where((mdl_errs > 1e-3))[0])/len(mdl_errs):.1f}%", fontsize=12, va='center', color='black')
    ax.text(0-1.75, -3.5-0.1, f"{100*len(np.where((mdl_errs > 1e-4) & (mdl_errs < 1e-3))[0])/len(mdl_errs):.1f}%", fontsize=12, va='center', color='black')
    ax.text(0-1.75, np.log10(1e-4)-0.4, f"{100*len(np.where(mdl_errs < 1e-4)[0])/len(mdl_errs):.1f}%", fontsize=12, va='center', color='black')

    colored_violinplot(ax, np.log10(cda_errs), positions=[0.9], colors=['none'], widths=0.8, border_colors=['k'])
    ax.scatter([0.9], [np.log10(np.mean(cda_errs))], color='k', label="Mean CDA Error")
    ax.text(0.9+0.2, np.log10(1e-3)+1.0, f"{100*len(np.where((cda_errs > 1e-3))[0])/len(cda_errs):.1f}%", fontsize=12, va='center', color='black')
    ax.text(0.9+0.1, -3.3, f"{100*len(np.where((cda_errs > 1e-4) & (cda_errs < 1e-3))[0])/len(cda_errs):.1f}%", fontsize=12, va='center', color='black')
    ax.text(0.9+0.15, np.log10(1e-4)-2.0, f"{100*len(np.where(cda_errs < 1e-4)[0])/len(cda_errs):.1f}%", fontsize=12, va='center', color='black')

    colored_violinplot(ax, np.log10(mdl_errs_nse), positions=[4.75], colors=['gray'], widths=0.8, border_colors=['gray'])
    ax.scatter([4.75], [np.log10(np.mean(mdl_errs_nse))], color='gray', label="Mean Model Error")
    ax.text(4.75-1.2, np.log10(1e-3)+1.0, f"{100*len(np.where((mdl_errs_nse > 1e-3))[0])/len(mdl_errs_nse):.1f}%", fontsize=12, va='center', color='gray')
    ax.text(4.75-1.75, -3.7, f"{100*len(np.where((mdl_errs_nse > 1e-4) & (mdl_errs_nse < 1e-3))[0])/len(mdl_errs_nse):.1f}%", fontsize=12, va='center', color='gray')
    ax.text(4.75-1.5, np.log10(1e-4)-0.5, f"{100*len(np.where(mdl_errs_nse < 1e-4)[0])/len(mdl_errs_nse):.1f}%", fontsize=12, va='center', color='gray')

    colored_violinplot(ax, np.log10(cda_errs_nse), positions=[5.7], colors=['gray'], widths=0.8, border_colors=['gray'])
    ax.scatter([5.7], [np.log10(np.mean(cda_errs_nse))], color='gray', label="Mean CDA Error")
    ax.text(5.7+0.2, np.log10(1e-3)+1.0, f"{100*len(np.where((cda_errs_nse > 1e-3))[0])/len(cda_errs_nse):.1f}%", fontsize=12, va='center', color='gray')
    ax.text(5.7+0.45, -3.7, f"{100*len(np.where((cda_errs_nse > 1e-4) & (cda_errs_nse < 1e-3))[0])/len(cda_errs_nse):.1f}%", fontsize=12, va='center', color='gray')
    ax.text(5.7+0.1, np.log10(1e-4)-0.9, f"{100*len(np.where(cda_errs_nse < 1e-4)[0])/len(cda_errs_nse):.1f}%", fontsize=12, va='center', color='gray')

    # First row: labels at individual x positions
    ax.xaxis.set_major_locator(FixedLocator([-0.2, 1.0, 4.65, 5.79]))
    ax.xaxis.set_major_formatter(FixedFormatter(['$\\mathbf{*}$', '[8]', '$\\mathbf{*}$', '[8]']))
    # rotate the labels for better readability
    ax.tick_params(axis='x', which='major', rotation=45)

    # Padding controls vertical separation between rows
    ax.tick_params(axis="x", which="major", pad=4, length=5, labelsize=12)
    ax.tick_params(axis="x", which="minor", pad=25, length=0, labelsize=15)


    ax.axhline(np.log10(1e-3), color='red', linestyle='--', alpha=0.3)
    ax.axhline(np.log10(1e-4), color='green', linestyle='--', alpha=0.3)
    ax.set_xlim(-1.75, 7.7)

    ax.axvline(2.9, color='black', linestyle='--', alpha=0.3)

    ax.set_title(f"({chr(65 + i).lower()})", loc='left', fontsize=16)

axs[-1].legend(
    handles=[Rectangle(
                (0, 0), 1, 1,
                facecolor="lightgrey",
                edgecolor="none",
                
                label="$\\pm \\sqrt{M}$"
            ),
            Rectangle(
                (0, 0), 1, 1,
                facecolor="none",
                edgecolor="black",
                label="0"
            )],
    handler_map={Rectangle: HandlerRectWithCircle()},
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0,
    title="Noise Level", title_fontsize=12, fontsize=12
    )


fig.tight_layout()

print_comparison_table(df)

fig.savefig("fig-11.pdf", bbox_inches='tight', dpi=300)

df.to_csv("stats-11.csv", index=False)

