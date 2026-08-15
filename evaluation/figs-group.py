import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from matplotlib.ticker import MultipleLocator
#import Markdown
from IPython.display import display, Markdown, HTML

import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.legend_handler import HandlerPatch

from matplotlib.patches import Patch
import pandas as pd

thresh_success = 1e-4
thresh_fail = 1e-3

ENERGIES = np.linspace(0.001, 1.0, 100)

def plotvio(vd, pos, noise=True, logscale=True, ax=None, text1=True, median=False):
    if ax is None:
        ax = plt.gca()

    plot_data = np.log10(vd) if logscale else vd

    # Disable matplotlib's automatic median
    parts = ax.violinplot(
        plot_data,
        positions=[pos],
        widths=0.9,
        showmedians=False
    )

    if noise:
        color = "gray"
        edgecolor = "gray"
        txtcolor = "gray"
    else:
        color = "white"
        edgecolor = "black"
        txtcolor = "black"

    mean = np.mean(vd)
    mean_y = np.log10(mean) if logscale else mean

    if noise:
        ax.scatter(
            pos, mean_y,
            color=edgecolor,
            s=100,
            zorder=6
        )
    else:
        ax.scatter(
            pos, mean_y,
            facecolors="none",
            edgecolors=edgecolor,
            s=100,
            zorder=10
        )

    if text1:
        ax.text(
            pos,
            mean_y + (1 if noise else -1) * (0.2 if logscale else 0.0002),
            f"{mean:.2e}",
            color=txtcolor,
            fontsize=10,
            ha="center",
            zorder=7
        )

    # Manual median
    if median:
        med = np.median(vd)
        med_y = np.log10(med) if logscale else med

        ax.hlines(
            med_y,
            pos - 0.2,
            pos + 0.2,
            color=edgecolor,
            linewidth=2,
            zorder=8
        )

    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor(edgecolor if not noise else "none")
        pc.set_alpha(0.5 if noise else 1.0)

    parts["cbars"].set_edgecolor(edgecolor)
    parts["cmins"].set_edgecolor(edgecolor)
    parts["cmaxes"].set_edgecolor(edgecolor)

    return np.mean(vd)

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

def print_stats_table(df_stats):
    flux_labels = {
        "0.01-1.0": "[*/100, *]",
        "1.0-1.0": "*",
        "0.1-0.1": "*/10",
        "0.01-0.01": "*/100",
    }

    flux_order = [
        "0.01-1.0",
        "1.0-1.0",
        "0.1-0.1",
        "0.01-0.01",
    ]

    train_order = ["Ideal", "A", "B"]
    noise_order = ["0.0-0.0", "0.0-1.0"]

    print("\nReconstruction errors (x10^-3)\n")

    header1 = (
        f"{'Test flux':<16}"
        f"{'Train':<10}"
        f"{'Test noise = 0':^30}"
        f"{'Test noise = [-1,1]':^30}"
    )

    header2 = (
        f"{'':<16}"
        f"{'':<10}"
        f"{'Mean':>10}"
        f"{'Median':>10}"
        f"{'IQR':>10}"
        f"{'Mean':>10}"
        f"{'Median':>10}"
        f"{'IQR':>10}"
    )

    print(header1)
    print(header2)
    print("-" * 86)

    for test_flux in flux_order:
        dff = df_stats[df_stats["test_flux"] == test_flux]

        for i, train_condition in enumerate(train_order):
            vals = []

            for test_noise in noise_order:
                row = dff[
                    (dff["train_condition"] == train_condition)
                    & (dff["test_noise"] == test_noise)
                ].iloc[0]

                vals.extend([
                    row["mean"] * 1e3,
                    row["median"] * 1e3,
                    row["iqr"] * 1e3,
                ])

            flux_label = flux_labels[test_flux] if i == 0 else ""

            print(
                f"{flux_label:<16}"
                f"{train_condition:<10}"
                f"{vals[0]:>10.3f}"
                f"{vals[1]:>10.3f}"
                f"{vals[2]:>10.3f}"
                f"{vals[3]:>10.3f}"
                f"{vals[4]:>10.3f}"
                f"{vals[5]:>10.3f}"
            )

        print("-" * 86)

def print_shape_flux_table_terminal(df_stats):
    noise_labels = {
        "0.0-0.0": "0",
        "0.0-1.0": "[-1,1]",
    }

    noise_order = ["0.0-0.0", "0.0-1.0"]
    train_order = ["Ideal", "A", "B"]

    metric_order = ["djs", "fae"]

    print("\nShape and flux reconstruction errors (x10^-2)\n")

    header1 = (
        f"{'Test noise':<14}"
        f"{'Train':<10}"
        f"{'Shape error':^30}"
        f"{'Flux error':^30}"
    )

    header2 = (
        f"{'':<14}"
        f"{'':<10}"
        f"{'Mean':>10}"
        f"{'Median':>10}"
        f"{'IQR':>10}"
        f"{'Mean':>10}"
        f"{'Median':>10}"
        f"{'IQR':>10}"
    )

    print(header1)
    print(header2)
    print("-" * 84)

    for test_noise in noise_order:
        dfn = df_stats[df_stats["test_noise"] == test_noise]

        for i, train_condition in enumerate(train_order):
            vals = []

            for metric_type in metric_order:
                row = dfn[
                    (dfn["train_condition"] == train_condition)
                    & (dfn["metric_type"] == metric_type)
                ].iloc[0]

                vals.extend([
                    row["mean"] * 1e2,
                    row["median"] * 1e2,
                    row["iqr"] * 1e2,
                ])

            noise_label = noise_labels[test_noise] if i == 0 else ""

            print(
                f"{noise_label:<14}"
                f"{train_condition:<10}"
                f"{vals[0]:>10.3f}"
                f"{vals[1]:>10.3f}"
                f"{vals[2]:>10.3f}"
                f"{vals[3]:>10.3f}"
                f"{vals[4]:>10.3f}"
                f"{vals[5]:>10.3f}"
            )

        print("-" * 84)


def plot_per_flux_tot_seed(erange, design, spec_type, model_full_name, metric_type, lim_type, test_fluxes=["0.01-1.0", "1.0-1.0", "0.1-0.1", "0.01-0.01"], seeds = [0, 1, 10, 33, 42], scale='log10', text=False, floor=None, good=False, bad=False):

    df_stats = pd.DataFrame(columns=["train_condition", "test_flux", "test_noise", "std", "min", "max", "mean", "median", "iqr"])

    train_conditions = {
        ("0.0001-65535.0-0.0001", "1.0-1.0",  "0.0-0.0"): "Ideal",
        ("10-65535-1",            "0.01-1.0", "0.0-0.0"): "A",
        ("10-65535-1",            "0.01-1.0", "0.0-1.0"): "B",
    }

    met_sym = {"mae": "$\epsilon_S$", "djs": "$\epsilon_{\\bar{S}}$", "fae": "$\epsilon_{\phi}$"}
    fig = plt.figure(figsize=(14, 4))
    ax = [None] * 4
    # Define grid: 3 rows, 2 columns
    gs = gridspec.GridSpec(1, 5, figure=fig)
    # First row: two plots
    ax[0] = fig.add_subplot(gs[0, 0:2])  # row 0, col 0
    ax[1] = fig.add_subplot(gs[0, 2], sharey=ax[0] )  # row 0, col 1
    # Second row: one plot spanning both columns
    ax[2] = fig.add_subplot(gs[0, 3], sharey=ax[0])  # row 1, col 0
    ax[3] = fig.add_subplot(gs[0, 4], sharey=ax[0])  # row 1, col 1
    
    no_ps = []
    nse_ps = []

    for a in ax[1:]:
        if a is not None:
            a.tick_params(
                axis='y',
                which='both',
                left=True,      # hide tick marks
                labelleft=False  # hide tick labels
            )
    
    for idx, test_flux in enumerate(test_fluxes):
        for j, test_noise in enumerate(["0.0-0.0", "0.0-1.0"]):
            sumrs = []
            # print(f"Test flux: {test_flux}, Test noise: {test_noise}:")
            for i, (train_b, train_flux, train_noise) in enumerate([("0.0001-65535.0-0.0001", "1.0-1.0",  "0.0-0.0"),
                                                                                ("10-65535-1",            "0.01-1.0", "0.0-0.0"),
                                                                                ("10-65535-1",            "0.01-1.0", "0.0-1.0"),
                                                                                ]):
                vds = []
                for seed in seeds:
                    vd = np.load(
                            f'../testing/{train_b}/{erange}/{design}/{spec_type}/'
                            f'noise--{train_noise}/flux--{train_flux}/{model_full_name}/{seed}/'
                            f'res/{metric_type}--B-10-65535-1--flux-{test_flux}--noise-{test_noise}--{lim_type}.npy'
                        ).clip(min=floor if floor is not None else None)
                    # remove nans and infs
                    # vd = vd[~np.isnan(vd)]
                    vds.append(vd)
                vd_tot = np.concatenate(vds)
                # if metric_type == 'mae':
                #     vd_tot = vd_tot * 100
                if idx != 0:
                    sumrs.append(plotvio(vd_tot, i, noise=j, ax=ax[idx], text1=text, logscale=scale=='log10'))
                else:
                    # plot next to each other
                    sumrs.append(plotvio(vd_tot, (i*3) + j +1 , noise=j, ax=ax[idx], text1=text, logscale=scale=='log10'))
                    ax[idx].text(i*3 + j + 1, np.log10(1e-3)+0.6, f"{100*len(np.where((vd_tot > 1e-3))[0])/len(vd_tot):.1f}%", fontsize=12, va='center', color='black' if j == 0 else 'gray')
                    ax[idx].text(i*3 + j + 1, -3.7, f"{100*len(np.where((vd_tot > 1e-4) & (vd_tot < 1e-3))[0])/len(vd_tot):.1f}%", fontsize=12, va='center', color='black' if j == 0 else 'gray')
                    ax[idx].text(i*3 + j + 1, np.log10(1e-4)-0.6, f"{100*len(np.where(vd_tot < 1e-4)[0])/len(vd_tot):.1f}%", fontsize=12, va='center', color='black' if j == 0 else 'gray')
                    if j == 1 and i != 0:
                        ax[idx].hlines(np.log10(sumrs[-1]) if scale == 'log10' else sumrs[-1], i*3 + j + 1, xmax=6.25, color='gray', linestyle='--', zorder=3)
                        nse_ps.append(sumrs[-1])
                        
                    # elif j == 0 and i != 0:
                    #     if i == 1:
                    #         ax[idx].hlines(np.log10(sumrs[-1]) if scale == 'log10' else sumrs[-1], i*2.5 + j, xmax=4, color='k', linestyle='--', zorder=3)
                    #     else:
                    #         ax[idx].hlines(np.log10(sumrs[-1]) if scale == 'log10' else sumrs[-1], 4, xmax=i*2.5 + j, color='k', linestyle='--', zorder=3)
                    #     no_ps.append(sumrs[-1])
                # print(f"Train flux: {train_flux}, Train noise: {train_noise}, Mean: {np.mean(vd_tot):.2e}, Median: {np.median(vd_tot):.2e}, IQR: {np.percentile(vd_tot, 75) - np.percentile(vd_tot, 25):.2e}, Std: {np.std(vd_tot):.2e}, Min: {np.min(vd_tot):.2e}, Max: {np.max(vd_tot):.2e}")
                df_stats.loc[len(df_stats)] = [f"{train_conditions.get((train_b, train_flux, train_noise), 'Unknown')}", test_flux, test_noise, np.std(vd_tot), np.min(vd_tot), np.max(vd_tot),  np.mean(vd_tot), np.median(vd_tot), np.percentile(vd_tot, 75) - np.percentile(vd_tot, 25)]
            
            for i in range(len(sumrs) -1):
                # ax[idx].plot([i, i+1], [np.log10(sumrs[i]), np.log10(sumrs[i+1])], linestyle='--' if j == 0 else '-', zorder=4, c="k" if j == 0 else "C0")
                if text:
                    ax[idx].text((i + i+1)/2, (np.log10(sumrs[i]) + np.log10(sumrs[i+1]))/2 + (-0.15 if j == 0 else 0.15), f"{sumrs[i+1]-sumrs[i]:.2e}", color="k" if j == 0 else "C0", fontsize=10, ha='center', zorder=7)

        
        ax[idx].set_xticklabels([])
    
    # # plot an arrow from p[0] and p[1] in ax[0]
    ax[0].annotate('', xy=(6.25, np.log10(nse_ps[1])-0.05 if scale == 'log10' else nse_ps[1]), xytext=(6.25, np.log10(nse_ps[0]) if scale == 'log10' else nse_ps[0]), arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    # ax[0].annotate('', xy=(4, np.log10(no_ps[1])+0.05 if scale == 'log10' else no_ps[1]), xytext=(4, np.log10(no_ps[0]) if scale == 'log10' else no_ps[0]), arrowprops=dict(arrowstyle='->', color='k', lw=2))

    # ax[0].text(6.5, np.log10(nse_ps[1]) + 0.5 if scale == 'log10' else nse_ps[1] + 0.1, f"$\Delta$={(nse_ps[1]-nse_ps[0])*1e4:.2f}$\\times10^{{-4}}$", color='gray', fontsize=12, ha='center')
    # ax[0].text(4, np.log10(no_ps[0]) - 0.5 if scale == 'log10' else no_ps[0] - 0.1, f"$\Delta$={(no_ps[1]-no_ps[0])*1e4:.2f}$\\times10^{{-4}}$", color='k', fontsize=12, ha='center')

    ax[0].set_title(f'Test $\phi \in [\phi^* / 100 , \phi^*]$', fontsize=16)
    ax[1].set_title(f' Test $\phi = \phi^*$', fontsize=16)
    ax[2].set_title(f'   Test $\phi = \phi^*/10$', fontsize=16)
    ax[3].set_title(f'      Test $\phi = \phi^*/100$', fontsize=16)
    
    ax[0].set_ylabel(f'log10 ({met_sym.get(metric_type, metric_type.upper())})' if scale == 'log10' else f'{metric_type.upper()}', fontsize=16)
    ax[0].tick_params(axis='y', labelsize=14)


    for i, axi in enumerate(ax):
        axi.set_xticks([0,1,2])
        axi.set_xticklabels(["Ideal", "A", "B"], fontsize=12)
        axi.set_xlabel(f"Training Conditions", fontsize=14)
        axi.set_title(f"({chr(65 + i).lower()})", loc='left', fontsize=16)
        # add a grid
        axi.grid(axis='y', linestyle='--', alpha=0.5)
        
    ax[0].set_xticks([1.5, 4.5, 7.5])
    ax[0].set_xticklabels(["Ideal", "A", "B"], fontsize=12)
    ax[-1].legend(
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
    bbox_to_anchor=(1.02, 0.5),  # x > 1 puts it outside the right edge
    borderaxespad=0,
    title="Noise Level", title_fontsize=12, fontsize=12
    )

    for ax in ax:
        if good:
            ax.axhline(np.log10(thresh_success) if scale == 'log10' else thresh_success, c="green", linestyle='--', linewidth=2, alpha=0.3)
        if bad:
            ax.axhline(np.log10(thresh_fail) if scale == 'log10' else thresh_fail, c="red", linestyle='--', linewidth=2, alpha=0.3)

    fig.tight_layout()

    print_stats_table(df_stats)

    return fig, ax


def plot_per_mtrc_tot_seed(erange, design, spec_type, model_full_name, lim_type, metric_ts, seeds = [0, 1, 10, 33, 42], scales='log10', text=False, floors=[None, None, None], ceils=[None, None, None], good=False, bad=False):
    met_sym = {"mae": "$\epsilon_S$", "djs": "$\epsilon_{\\bar{S}}$", "fae": "$\epsilon_{\phi}$"}
    met_ttl = {"mae": "Reconstruction Error", "djs": "Shape Error", "fae": "Flux Error"}
    fig, ax = plt.subplots(1, len(metric_ts), figsize=(15, 4))
    scales = [scales] * len(metric_ts) if isinstance(scales, str) else scales

    df_stats = pd.DataFrame(columns=["train_condition", "test_noise", "metric_type", "std", "min", "max", "mean", "median", "iqr"])

    train_conditions = {
        ("0.0001-65535.0-0.0001", "1.0-1.0",  "0.0-0.0"): "Ideal",
        ("10-65535-1",            "0.01-1.0", "0.0-0.0"): "A",
        ("10-65535-1",            "0.01-1.0", "0.0-1.0"): "B",
    }
    
    for idx, metric_type in enumerate(metric_ts):
        floor = floors[idx]
        ceil = ceils[idx]
        nse_ps = []
        for j, test_noise in enumerate(["0.0-0.0", "0.0-1.0"]):
            sumrs = []
            for i, (train_b, train_flux, train_noise) in enumerate([("0.0001-65535.0-0.0001", "1.0-1.0",  "0.0-0.0"),
                                                                                ("10-65535-1",            "0.01-1.0", "0.0-0.0"),
                                                                                ("10-65535-1",            "0.01-1.0", "0.0-1.0"),
                                                                                ]):
                vds = []
                for seed in seeds:
                    vd = np.load(
                            f'../testing/{train_b}/{erange}/{design}/{spec_type}/'
                            f'noise--{train_noise}/flux--{train_flux}/{model_full_name}/{seed}/'
                            f'res/{metric_type}--B-10-65535-1--flux-0.01-1.0--noise-{test_noise}--{lim_type}.npy'
                        )
                    # remove nans and infs
                    vd = vd[~np.isnan(vd)]
                    vds.append(vd)
                vd_tot = np.concatenate(vds)
                # print(f"metric {metric_type}, trained with {train_flux}, {train_noise}, test noise {test_noise}: mean {np.mean(vd_tot):.5e}, max {np.max(vd_tot):.5e}, min {np.min(vd_tot):.5e}")
                df_stats.loc[len(df_stats)] = [f"{train_conditions.get((train_b, train_flux, train_noise), 'Unknown')}", test_noise, metric_type, np.std(vd_tot), np.min(vd_tot), np.max(vd_tot),  np.mean(vd_tot), np.median(vd_tot), np.percentile(vd_tot, 75) - np.percentile(vd_tot, 25)]
                # if metric_type == 'mae':
                #     vd_tot = vd_tot * 100
                sumrs.append(plotvio(vd_tot, i, noise=j, ax=ax[idx], text1=text, logscale=scales[idx]=='log10'))
                if j == 1 and i != 0:
                        ax[idx].hlines(np.log10(sumrs[-1]), i, xmax=2.5, color='gray', linestyle='--', zorder=3)
                        nse_ps.append(sumrs[-1])
            
            for i in range(len(sumrs) -1):
                # ax[idx].plot([i, i+1], [np.log10(sumrs[i]), np.log10(sumrs[i+1])], linestyle='--' if j == 0 else '-', zorder=4, c="k" if j == 0 else "C0")
                if text:
                    ax[idx].text((i + i+1)/2, (np.log10(sumrs[i]) + np.log10(sumrs[i+1]))/2 + (-0.15 if j == 0 else 0.15), f"{sumrs[i+1]-sumrs[i]:.2e}", color="k" if j == 0 else "C0", fontsize=10, ha='center', zorder=7)

        ax[idx].annotate('', xy=(2.5, np.log10(nse_ps[1])-0.05 if scales[idx] == 'log10' else nse_ps[1]), xytext=(2.5, np.log10(nse_ps[0]) if scales[idx] == 'log10' else nse_ps[0]), arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        
        ax[idx].set_xticks([0,1,2])
        ax[idx].set_xticklabels(["Ideal", "A", "B"], fontsize=12)
        ax[idx].tick_params(axis='y', labelsize=14)
        ax[idx].set_ylabel(f'log10 ( {met_sym.get(metric_type, metric_type.upper())} )' if scales[idx] == 'log10' else f'{met_sym.get(metric_type, metric_type.upper())}', fontsize=14)
        ax[idx].set_xlabel('Training Conditions', fontsize=14)
        ax[idx].set_ylim([np.log10(floor) if scales[idx] == 'log10' and floor is not None else floor if floor is not None else None, np.log10(ceil) if scales[idx] == 'log10' and ceil is not None else ceil if ceil is not None else None])
        ax[idx].set_title(met_ttl.get(metric_type, metric_type.upper()), fontsize=16)
        ax[idx].set_title(f"({chr(65 + idx).lower()})", loc='left', fontsize=16)

    if good:
        ax[0].axhline(np.log10(thresh_success) if scales[i] == 'log10' else thresh_success, c="green", linestyle='--', linewidth=2, alpha=0.3)
    if bad:
        ax[0].axhline(np.log10(thresh_fail) if scales[i] == 'log10' else thresh_fail, c="red", linestyle='--', linewidth=2, alpha=0.3)

    ax[-1].legend(
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
    bbox_to_anchor=(1.02, 0.5),  # x > 1 puts it outside the right edge
    borderaxespad=0,
    title="Noise Level", title_fontsize=12, fontsize=12
    )

    fig.tight_layout()

    print_shape_flux_table_terminal(df_stats)

    df_stats.to_csv(f"stats-12.csv", index=False)

    return fig, ax


def plot_all_eval(
    erange,
    designs,
    spec_types,
    spec_tests,
    model_full_name,
    train_noise,
    test_noise,
    metric_type,
    lim_type,
    b,
    train_flux,
    test_flux = '0.01-1.0',
    seeds = [0, 1, 10, 33, 42],
    scale='log10',
    thresh_success=None,
    thresh_fail=None,
    floor=None,
):

    colors = {
    'boltzmann': 'C5',
    'gaussian': 'C1',
    'skew': 'C4',
    'shighorder': 'C0',
    }

    test_labels = {  
        'boltzmann': 'B',
        'gaussian': 'G',
        'skew': 'SG',
        'shighorder': '  HO',
    }

    train_labels = {  
        'boltzmann': 'Boltzmann',
        'gaussian': 'Gaussian',
        'skew': 'Skewed Gaussian',
        'shighorder': 'High Order',
    }

    design_labels = {
        'existing': "existing",
        'lowe': "low energy",
        "myth-10-f-d": "idealised"
    }

    fig, axs = plt.subplots(1, len(designs), figsize=(9, 4), sharey=True)
    axs = np.atleast_1d(axs)
    

    if type(erange) is str:
        erange = [erange] * len(designs)

    if type(train_flux) is str:
        train_flux = [train_flux] * len(designs)

     # Fixed test flux level for evaluation

    for a, design in enumerate(designs):
        ax = axs[a]
        c = 0
        for i, s in enumerate(spec_tests):
            ax.axvspan(c, c + len(s) + 1, color=colors.get(spec_types[i], 'C0'), alpha=0.3)
            c += len(s) + 1

        c = 1
        x_ticks = []
        x_tick_labels = []
        
        # thresholds as horizontal lines
        
        
        for i, spec in enumerate(spec_types):
            for j, spec_test in enumerate(spec_tests[i]):
                try:
                    vio_datas = []
                    for seed in seeds:
                        
                        try:
                            vio_path = (
                                f'../testing/{b}/{erange[a]}/{design}/{spec}/'
                                f'noise--{train_noise}/flux--{train_flux[a]}/{model_full_name}/{seed}/'
                                f'res/{metric_type}--B-{b}--flux-{test_flux}--noise-{test_noise}--{lim_type}--{spec_test}.npy'
                            )
                            # print(f"Trying to load: {vio_path}")
                            vio_data = np.load(vio_path).clip(min=floor) if floor is not None else np.load(vio_path)
                        except:
                            if spec_test == spec:
                                vio_path = (f'../testing/{b}/{erange[a]}/{design}/{spec}/'
                                f'noise--{train_noise}/flux--{train_flux[a]}/{model_full_name}/{seed}/'
                                f'res/{metric_type}--B-{b}--flux-{test_flux}--noise-{test_noise}--{lim_type}.npy'
                                
                            )
                                # print(f"Trying to load: {vio_path}")
                                vio_data = np.load(vio_path).clip(min=floor) if floor is not None else np.load(vio_path)
                            else:
                                print(f"Could not load: {vio_path}")
                                continue
                        vio_datas.append(vio_data)
                    vio_data = np.concatenate(vio_datas).flatten()
                    # if metric_type == 'mae':
                    #     vio_data = vio_data * 100
                    # transform for plotting
                    plot_vals = np.log10(vio_data) if scale == 'log10' else vio_data

                    parts = ax.violinplot(
                        plot_vals, positions=[c], showmeans=False, showmedians=False, widths=0.8
                    )

                    for pc in parts['bodies']:
                        pc.set_facecolor(colors.get(spec_test, 'C0'))
                        pc.set_edgecolor(None)
                        pc.set_alpha(0.7)
                    parts['cbars'].set_edgecolor("black")
                    parts['cmins'].set_edgecolor("black")
                    parts['cmaxes'].set_edgecolor("black")

                    # scatter mean
                    mean_val = np.mean(vio_data)
                    std_val = np.std(vio_data)
                    scat_y = np.log10(mean_val) if scale == 'log10' else mean_val
                    ax.scatter(
                        c, scat_y, color="black", s=100,
                        label=f"{mean_val:.2e}$\\pm${std_val:.2e}",
                        zorder=5
                    )

                    # record x-ticks
                    x_ticks.append(c)
                    x_tick_labels.append(f'{test_labels.get(spec_test, spec_test)}')
                    
                    c += 1

                except:
                    # print(f"Could not load: {vio_path}")
                    pass
            
            c += 1

        # prettify axes after iterating train_flux~
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        # --- Top ticks (secondary axis) ---
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks([0.5 + len(spec_tests[i])//2 + sum(len(spec_tests[j]) + 1 for j in range(i)) for i in range(len(spec_tests))])
        ax_top.set_xticklabels([f"{train_labels.get(spe, spe)} Agent" for spe in spec_types], fontsize=12)
        # ax_top.set_xlabel('Trained Agent', fontsize=14)

        # ax.set_title(f"{design_labels.get(design, design)} design", fontsize=15)
        ax.set_title(f"Training Spectra", fontsize=14)

        ax.set_xlim(0, c-1)    
        ax.set_xlabel('Test Spectra', fontsize=14)
    if thresh_success is not None:
            ax.axhline(np.log10(thresh_success) if scale == 'log10' else thresh_success, c="green", linestyle='--', linewidth=2, alpha=0.3)
    if thresh_fail is not None:
            ax.axhline(np.log10(thresh_fail) if scale == 'log10' else thresh_fail, c="red", linestyle='--', linewidth=2, alpha=0.3)
    axs[0].set_ylabel(f'log10 ($\epsilon_S$)', fontsize=14)
    fig.tight_layout()

    return fig, axs


fig, ax = plot_per_flux_tot_seed(
            erange="0.001-1.0--100--5.0",
            design="existing",
            spec_type='shighorder',
            model_full_name='mlp--20-lin--10-log--10x',
            metric_type='mae',
            lim_type='our',
            scale='log10',
            seeds = [0, 1, 10, 33, 42],
            text=False,
            good=True,
            bad=True
        )

fig.savefig("fig-8.pdf", bbox_inches='tight', dpi=300)

fig, ax = plot_per_mtrc_tot_seed(
            erange="0.001-1.0--100--5.0",
            design="existing",
            spec_type='shighorder',
            model_full_name='mlp--20-lin--10-log--10x',
            lim_type='our',
            scales=['log10', 'log10', 'log10'],
            metric_ts=['mae', 'djs', 'fae'],
            seeds = [0, 1, 10, 33, 42],
            text=False,
            floors=[None, 1e-7, 5e-5],
            ceils = [None, None, None],
            good=True,
            bad=True
        )

fig.savefig("fig-13.pdf", bbox_inches='tight', dpi=300)

fig, ax = plot_all_eval(
    erange='0.001-1.0--100--5.0',
    designs=["existing"],
    spec_types=['boltzmann', 'gaussian', 'skew', 'shighorder'],
    spec_tests=[['boltzmann', 'gaussian', 'skew', 'shighorder'],
                ['boltzmann', 'gaussian', 'skew', 'shighorder'],
                ['boltzmann', 'gaussian', 'skew', 'shighorder'],
                ['boltzmann', 'gaussian', 'skew', 'shighorder']],
    model_full_name='mlp--20-lin--10-log--10x',
    train_noise="0.0-1.0",
    test_noise="0.0-1.0",
    metric_type='mae',
    lim_type='our',
    b='10-65535-1',
    train_flux='0.01-1.0',
    seeds=[0, 1, 10, 33, 42],
    scale='log10',
    floor=1e-5,
    thresh_success=thresh_success,
    thresh_fail=thresh_fail
)

fig.savefig("fig-9.pdf", bbox_inches='tight', dpi=300)