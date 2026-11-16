import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm
from features_library import *
from models import get_model, latex_labels


def plot_histogram_comparison(
    results_dir,
    theta,
    theta_gt,
    bins=15,
    legends_name=("Pred,", "GT,"),
    xlabel="Value",
    ylabel="Density",
    filename="histogram",
    label_fs=20,
    # title_fs=20,
    tick_fs=20,
    legend_fs=12
):
    os.makedirs(results_dir, exist_ok=True)

    theta = np.asarray(theta).reshape(-1)
    
    color_gt   = "#0072B2"   # blue
    color_pred = "#D55E00"   # vermillion/orange

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    if theta_gt is not None:
        theta_gt = np.asarray(theta_gt).reshape(-1)
        ax.hist(
            theta_gt,
            bins=bins,
            density=True,
            histtype="step",
            color=color_gt,
            linewidth=2.0,
            label=legends_name[1]
        )

    ax.hist(
        theta,
        bins=bins,
        density=True,
        alpha=0.55,
        color=color_pred,
        edgecolor="black",
        linewidth=0.4,
        label=legends_name[0]
    )

    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)

    ax.tick_params(axis="both", labelsize=tick_fs)

    ax.legend(
        frameon=True,
        fontsize=legend_fs,
        # loc="upper left",
        handlelength=1.0,
        borderpad=0.2,
        labelspacing=0.2
    )

    # ax.grid(True, alpha=0.25)
    ax.set_box_aspect(1)

    plt.tight_layout()

    base_name = os.path.splitext(filename)[0]

    png_path = os.path.join(results_dir, base_name + ".png")

    svg_dir = os.path.join(results_dir, "svg")
    os.makedirs(svg_dir, exist_ok=True)

    svg_path = os.path.join(svg_dir, base_name + ".svg")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved histogram: {png_path}")

def plot_hexbin_parity(
    results_dir,
    theta,
    theta_gt,
    legends_name=("1:1", "Hexbin count"),
    xlabel="Ground Truth",
    ylabel="Prediction",
    filename="hexbin_parity",
    label_fs=20,
    # title_fs=25,
    tick_fs=20,
    legend_fs=16,
    annot_fs=18,
    cbar_label_fs=18,
    cbar_tick_fs=18,
    cmap="viridis"
):
    os.makedirs(results_dir, exist_ok=True)

    theta = np.asarray(theta).reshape(-1)
    theta_gt = np.asarray(theta_gt).reshape(-1)

    valid = np.isfinite(theta) & np.isfinite(theta_gt)
    theta = theta[valid]
    theta_gt = theta_gt[valid]

    # R^2
    ss_res = np.sum((theta_gt - theta) ** 2)
    ss_tot = np.sum((theta_gt - np.mean(theta_gt)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(5.2, 4.5))

    min_val = min(theta_gt.min(), theta.min())
    max_val = max(theta_gt.max(), theta.max())
    pad = 0.05 * (max_val - min_val)
    lims = [min_val - pad, max_val + pad]

    hb = ax.hexbin(
        theta_gt,
        theta,
        gridsize=32,
        cmap=cmap,
        mincnt=1,
        norm=LogNorm(),
        linewidths=0.0
    )

    ax.plot(
        lims,
        lims,
        linestyle="--",
        color="black",
        linewidth=1.0,
        label=legends_name[0]
    )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)

    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    # ax.set_title(title, fontsize=title_fs)

    ax.tick_params(axis="both", labelsize=tick_fs)

    ax.text(
        0.05,
        0.95,
        rf"$R^2 = {r2:.4f}$",
        transform=ax.transAxes,
        fontsize=annot_fs,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9
        )
    )

    cbar = fig.colorbar(hb, ax=ax, fraction=0.040, pad=0.03)
    cbar.set_label(legends_name[1], fontsize=cbar_label_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    ax.legend(
        frameon=True,
        loc="lower right",
        fontsize=legend_fs
    )

    # ax.grid(True, alpha=0.20)

    plt.tight_layout()

    base_name = os.path.splitext(filename)[0]

    png_path = os.path.join(results_dir, base_name + ".png")
    
    svg_dir = os.path.join(results_dir, "svg")
    os.makedirs(svg_dir, exist_ok=True)

    svg_path = os.path.join(svg_dir, base_name + ".svg")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved hexbin parity: {png_path}")

    return r2

def plot_histogram_gmm(
    theta,
    results_dir,
    bins=50,
    n_modes=2,
    bar_color="#D55E00",
    gmm_color="#222222",
    comp_color="#0072B2",
    unit="MPa",
    params=None
):

    os.makedirs(results_dir, exist_ok=True)

    # --------------------------------------------------------
    # Convert theta to numpy
    # --------------------------------------------------------
    if hasattr(theta, "detach"):
        theta = theta.detach().cpu().numpy()

    theta = np.asarray(theta, dtype=np.float64).reshape(-1)

    # --------------------------------------------------------
    # Split theta into material fields
    # theta = [params of the model] per element, e.g. [C10] for NH
    # --------------------------------------------------------
    if params is None:
        params = get_model()['params']
    numFeatures = len(params)  # Number of features per material
    print(f"Number of features per material: {numFeatures}")

    if theta.shape[0] % numFeatures != 0:
        raise ValueError(
            f"theta length = {theta.shape[0]} is not divisible by numFeatures = {numFeatures}"
        )

    idxs = np.arange(theta.shape[0] // numFeatures)

    fields = {p: theta[numFeatures*idxs + j] for j, p in enumerate(params)}

    # Latex labels: see models.py

    label_map = {
        1: "Unimodal Distribution Fit",
        2: "Bimodal Distribution Fit",
        3: "Trimodal Distribution Fit"
    }

    gmm_label = label_map.get(n_modes, f"Fitted {n_modes}-mode GMM")

    # --------------------------------------------------------
    # Plot formatting consistent with previous figures
    # --------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,
        "mathtext.fontset": "dejavuserif",
    })

    # --------------------------------------------------------
    # Storage for CSV
    # --------------------------------------------------------
    rows_long = []

    theta_mean = np.zeros((n_modes, numFeatures), dtype=np.float64)
    theta_variance = np.zeros((n_modes, numFeatures), dtype=np.float64)

    field_names = list(fields.keys())

    # --------------------------------------------------------
    # Loop over each material field
    # --------------------------------------------------------
    for p_id, field_name in enumerate(field_names):

        values = np.asarray(fields[field_name], dtype=np.float64).reshape(-1)

        theta_label = latex_labels.get(field_name, rf"$\theta_{{{field_name}}}$")

        # ----------------------------------------------------
        # Fit GMM
        # ----------------------------------------------------
        gmm = GaussianMixture(
            n_components=n_modes,
            covariance_type="full",
            random_state=0
        )

        gmm.fit(values.reshape(-1, 1))

        weights = gmm.weights_.flatten()
        means = gmm.means_.flatten()
        variances = gmm.covariances_.reshape(n_modes, -1)[:, 0]

        # Sort by mean: material_id 1 = lower mean, material_id 2 = higher mean
        order = np.argsort(means)

        weights = weights[order]
        means = means[order]
        variances = variances[order]

        # Store mean and variance material-wise
        theta_mean[:, p_id] = means
        theta_variance[:, p_id] = variances

        # ----------------------------------------------------
        # Plot histogram + fitted GMM
        # ----------------------------------------------------
        # fig, ax = plt.subplots(figsize=(10, 6))
        fig, ax = plt.subplots(figsize=(11.5, 3.8))  # For 2D experimental cases

        ax.hist(
            values,
            bins=bins,
            alpha=0.55,
            density=True,
            color=bar_color,
            edgecolor="black",
            linewidth=0.5,
            label=rf"Prediction, {theta_label}"
        )

        # x = np.linspace(values.min(), values.max(), 500).reshape(-1, 1)

        # logprob = gmm.score_samples(x)
        # pdf = np.exp(logprob)

        # ax.plot(
        #     x.ravel(),
        #     pdf,
        #     linestyle="--",
        #     color=gmm_color,
        #     linewidth=2,
        #     label=gmm_label
        # )

        # # Plot individual sorted components
        # for i, (w, m, var) in enumerate(zip(weights, means, variances), start=1):

        #     comp_pdf = w * (1.0 / np.sqrt(2.0*np.pi*var)) * np.exp(
        #         -0.5 * (x.ravel() - m)**2 / var
        #     )

        #     ax.plot(
        #         x.ravel(),
        #         comp_pdf,
        #         linestyle=":",
        #         color=comp_color,
        #         linewidth=1.5
        #     )

        #     peak_y = w * (1.0 / np.sqrt(2.0*np.pi*var))

        #     ax.plot(m, peak_y, "ko")

        #     ax.text(
        #         m,
        #         peak_y * 1.08,
        #         rf"M{i}" + "\n" +
        #         rf"$\mu$={m:.4f}" + "\n" +
        #         rf"$\sigma^2$={var:.2e}" + "\n" +
        #         rf"$w$={w:.2f}",
        #         fontsize=9,
        #         ha="center",
        #         va="bottom",
        #         bbox=dict(
        #             boxstyle="round",
        #             facecolor="white",
        #             edgecolor="gray",
        #             alpha=0.85
        #         )
        #     )

        #     rows_long.append({
        #         "parameter": field_name,
        #         "material_id": i,
        #         "theta_mean": m,
        #         "theta_variance": var,
        #         "theta_std": np.sqrt(var),
        #         "weight": w
        #     })

        ax.set_xlabel(rf"{theta_label} ({unit})")
        ax.set_ylabel("Density")
        # ax.set_title(rf"Distribution of {theta_label}")
        ax.legend(frameon=True)
        # ax.legend(frameon=True, loc="center left")
        # ax.legend(
        #     frameon=True,
        #     loc="upper left",
        #     bbox_to_anchor=(0.02, 0.40),
        # )
        # ax.grid(True, alpha=0.25)

        plt.tight_layout()
        png_path = os.path.join(
            results_dir,
            f"theta_histogram_{field_name}_{n_modes}mode_bins{bins}.png"
        )

        svg_dir = os.path.join(results_dir, "svg")
        os.makedirs(svg_dir, exist_ok=True)

        svg_path = os.path.join(
            svg_dir,
            f"theta_histogram_{field_name}_{n_modes}mode_bins{bins}.svg"
        )

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")

        plt.close(fig)

        print(f"\n{field_name} -- {gmm_label}:")
        for i, (w, m, var) in enumerate(zip(weights, means, variances), start=1):
            print(
                f"  Material {i}: "
                f"weight={w:.4f}, mean={m:.6f}, variance={var:.6e}"
            )

    # --------------------------------------------------------
    # Save long CSV
    # --------------------------------------------------------
    df_long = pd.DataFrame(rows_long)

    long_csv_path = os.path.join(
        results_dir,
        "theta_mean_variance_materialwise_long.csv"
    )

    df_long.to_csv(long_csv_path, index=False)

    # --------------------------------------------------------
    # Save wide CSV
    # --------------------------------------------------------
    wide_rows = []

    for mat_id in range(1, n_modes + 1):

        row = {"material_id": mat_id}

        for p_id, field_name in enumerate(field_names):

            row[f"{field_name}_mean"] = theta_mean[mat_id-1, p_id]
            row[f"{field_name}_variance"] = theta_variance[mat_id-1, p_id]
            row[f"{field_name}_std"] = np.sqrt(theta_variance[mat_id-1, p_id])

        wide_rows.append(row)

    df_wide = pd.DataFrame(wide_rows)

    wide_csv_path = os.path.join(
        results_dir,
        "theta_mean_variance_materialwise_wide.csv"
    )

    df_wide.to_csv(wide_csv_path, index=False)

    print("\nSaved CSV files:")
    print(f"  {long_csv_path}")
    print(f"  {wide_csv_path}")

    print("\nSaved plots:")
    
    for field_name in field_names:
        print(f"  theta_histogram_{field_name}_{n_modes}mode.png")
        print(f"  theta_histogram_{field_name}_{n_modes}mode.svg")

    return theta_mean, theta_variance, df_long, df_wide


def plot_scatter_parity(
    results_dir,
    theta,
    theta_gt,
    legends_name=("Pred", "1:1 reference"),
    xlabel="Ground Truth",
    ylabel="Prediction",
    filename="scatter_parity",
    title="Prediction vs Ground Truth"
):
    os.makedirs(results_dir, exist_ok=True)

    theta = np.asarray(theta).reshape(-1)
    theta_gt = np.asarray(theta_gt).reshape(-1)

    # R^2
    ss_res = np.sum((theta_gt - theta) ** 2)
    ss_tot = np.sum((theta_gt - np.mean(theta_gt)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    color_pred = plt.cm.viridis(0.50)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    min_val = min(theta_gt.min(), theta.min())
    max_val = max(theta_gt.max(), theta.max())
    pad = 0.05 * (max_val - min_val)
    lims = [min_val - pad, max_val + pad]

    ax.scatter(
        theta_gt,
        theta,
        s=14,
        color=color_pred,
        alpha=0.75,
        edgecolor="none",
        label=legends_name[0]
    )

    ax.plot(
        lims,
        lims,
        linestyle="--",
        color="black",
        linewidth=1.5,
        label=legends_name[1]
    )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.text(
        0.05,
        0.95,
        rf"$R^2 = {r2:.4f}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9)
    )

    ax.legend(frameon=True, loc="lower right")
    # ax.grid(True, alpha=0.25)

    plt.tight_layout()

    base_name = os.path.splitext(filename)[0]
    
    png_path = os.path.join(results_dir, base_name + ".png")
    svg_path = os.path.join(results_dir, base_name + ".svg")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved scatter parity: {png_path}")

    return r2   


if __name__=='__main__':
	# The plotting functions need the predicted / ground-truth theta, they are called from main_fgm_HEX.py
	print("Usage: called from main_fgm_HEX.py (plot_histogram_comparison, plot_hexbin_parity, plot_histogram_gmm, plot_scatter_parity)")
