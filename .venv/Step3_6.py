#!/usr/bin/env python3
"""
Step 6: Full Grid Path-Loss Fitting and Analysis

  1) Generate received-power heatmap
  2) Extract 1D transect at emitter height and fit log-distance model
  3) Fit log-distance model on entire 2D grid
  4) Compute and display residual statistics and heatmaps
  5) Perform KS test on 1D residuals
  6) Compute link-reliability vs. range
"""

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import erfcinv
import multiprocessing

# Project modules
from environment import Environment
from raytracing import RayTracing
from position import Position
from receiver import Receiver
from heatmap import create_heatmap

# Output directory for plots
OUTDIR = r'C:/Users/alexb/OneDrive - Université Libre de Bruxelles/MA1-ULBDrive/ELEC-H415/ELEC-H415_V2V/Plot'
os.makedirs(OUTDIR, exist_ok=True)


def fit_path_loss(distances, losses_dB):
    """
    Fit the log-distance path-loss model:
      loss(d) = L0 + 10·n·log10(d)
    Returns intercept L0, exponent n, and R².
    """
    x = np.log10(distances)
    y = losses_dB
    slope, intercept, r_val, _, _ = stats.linregress(x, y)
    n = slope / 10
    return intercept, n, r_val**2


def compute_max_range_fitted(Ptx_dBm, Psens_dBm, margin_dB, G_TX, L0_d0, n):
    """
    Compute maximum range given fade margin using the fitted log-distance model.
    """
    budget = Ptx_dBm - Psens_dBm - margin_dB + 20*np.log10(G_TX)
    exponent = (budget - L0_d0) / (10 * n)
    return 10**exponent


def compute_max_range_Friis(Ptx_dBm, Psens_dBm, margin_dB, n, G_TX, freq_MHz):
    """
    Compute maximum range using the free-space (Friis) formula plus fade margin.
    """
    C = 20*np.log10(freq_MHz) - 147.5 - 20*np.log10(G_TX)
    budget = Ptx_dBm - Psens_dBm - margin_dB
    exp = (budget - C) / (10 * n)
    return 10**exp


def main():
    # Initialize environment and ray tracer
    env = Environment()
    env.emitters[0].position = Position(1, 10)
    rt = RayTracing(env)

    # Heatmap parameters
    resolution = 5
    width, height = 1000, 21

    # 1) Generate received-power heatmap (dBm)
    P_meas = create_heatmap(env, width, height, resolution)
    Ptx_dBm = 10 * np.log10(rt.P_TX / 1e-3)

    # Build grid coordinates and distance matrix
    xs = np.arange(0, width, resolution)
    ys = np.arange(0, height, resolution)
    X, Y = np.meshgrid(xs, ys)
    emitter = env.emitters[0].position
    D = np.hypot(X - emitter.x, Y - emitter.y)
    D[D == 0] = 1e-3  # avoid log10(0)

    # 2) 1D transect line at emitter y-coordinate
    row_idx = np.argmin(np.abs(ys - emitter.y))
    d_line = np.arange(0, width, resolution)
    x_pos = emitter.x + d_line
    idx = np.clip(np.round(x_pos / resolution).astype(int), 0, len(xs)-1)
    L_meas_line = Ptx_dBm - P_meas[row_idx, idx]
    L0_line = L_meas_line + 20*np.log10(rt.G_TX)
    mask_line = d_line > 0

    # Fit on 1D transect
    L0_d0_line, n_line, R2_line = fit_path_loss(d_line[mask_line], L0_line[mask_line])
    print(f"1D fit: L0(d0)={L0_d0_line:.2f} dB, n={n_line:.3f}, R²={R2_line:.3f}")

    # Plot 1D path-loss vs distance
    d_vals = np.linspace(0.1, width, 200)
    L_fit_line = L0_d0_line + 10*n_line*np.log10(d_vals)
    G = rt.G_TX
    L_friis = 20*np.log10(d_vals) + 20*np.log10(rt.frequency) - 147.55 - 20*np.log10(G)

    plt.figure(figsize=(8,5))
    plt.plot(d_line, L0_line, 'o', ms=4, label='Measured Path Loss')
    plt.plot(d_vals, L_fit_line, 'r-', lw=2,
             label=f'1D Fit: {L0_d0_line:.2f} + 10·{n_line:.3f}·log10(d)')
    plt.plot(d_vals, L_friis, 'k--', label='FSPL')
    plt.xlabel('Distance (m)')
    plt.ylabel('Path Loss (dB)')
    plt.title('Path Loss vs. Distance For Receiver Moving along centerline')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'pathloss_1d.png'), dpi=300)
    plt.show()

    # 3) Fit on full 2D grid
    L_meas_grid = Ptx_dBm - P_meas
    mask_grid = D > 0
    distances_flat = D[mask_grid].ravel()
    losses_flat = L_meas_grid[mask_grid].ravel()
    L0_d0_full, n_full, R2_full = fit_path_loss(distances_flat, losses_flat)
    print(f"2D fit: L0(d0)={L0_d0_full:.2f} dB, n={n_full:.3f}, R²={R2_full:.3f}")

    # Reconstruct fitted grid and compute residuals
    L_fit_grid = L0_d0_full + 10 * n_full * np.log10(D)
    residuals_grid = L_fit_grid - L_meas_grid
    var_res = np.sqrt(np.var(residuals_grid, ddof=1))
    print(f"Variance of residuals on full grid: {var_res:.2f} dB")

    # 2D path-loss curves: measured as line, fitted, Friis
    # 2D path-loss curves: measured as line, fitted, Friis
    idx2 = np.argsort(distances_flat)
    plt.figure(figsize=(8, 5))

    # Measured Path Loss as a continuous line (not markers)
    plt.plot(distances_flat[idx2], losses_flat[idx2],
             '-', lw=1, alpha=0.7, label='Measured Path Loss')

    # Log-Distance fit
    plt.plot(distances_flat[idx2],
             L0_d0_full + 10 * n_full * np.log10(distances_flat[idx2]),
             'r-', lw=2,
             label=f' Fitted Path Loss: {L0_d0_full:.2f}+10·{n_full:.3f}·log10(d)')

    # Friis Free-Space curve
    friis2 = (20 * np.log10(distances_flat[idx2]) +
              20 * np.log10(rt.frequency) -
              147.55 - 20 * np.log10(rt.G_TX))
    plt.plot(distances_flat[idx2], friis2,
             'k--', lw=2, label='FSPL')

    plt.xlabel('Distance (m)')
    plt.ylabel('Path Loss (dB)')
    plt.title(' Path Loss on the Full Street')
    plt.xlim(0, 1000)  # force axis to 0–1000 m
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'pathloss2d_curve.png'), dpi=300)
    plt.show()

    # Heatmap of fitted path-loss and residuals
    plt.figure(figsize=(8,5))
    pcm = plt.pcolormesh(X, Y, L_fit_grid, shading='auto', cmap='viridis')
    plt.colorbar(pcm, label='Fitted Path Loss (dB)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Log-Distance Fit on Full Street')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'pathloss_2d_fit.png'), dpi=300)
    plt.show()

    plt.figure(figsize=(8,5))
    pcm = plt.pcolormesh(X, Y, np.abs(residuals_grid), shading='auto', cmap='plasma')
    plt.colorbar(pcm, label='Absolute Residual (dB)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Residuals Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'residuals_heatmap.png'), dpi=300)
    plt.show()

    # 4) Residuals on 1D transect and KS test using 1D fit
    L0_pred_line = L0_d0_line + 10 * n_line * np.log10(d_line[mask_line])
    res_line = L0_line[mask_line] - L0_pred_line
    res_centered = res_line - res_line.mean()
    sigma_L = res_centered.std(ddof=1)
    print(f"Sigma_L (1D residuals): {sigma_L:.2f} dB")
    stat, pval = stats.kstest(res_centered, 'norm', args=(0, sigma_L))
    print(f"KS test: statistic={stat:.3f}, p-value={pval:.3f}")

    plt.figure(figsize=(8,5))
    plt.hist(res_centered, bins=50, density=True, alpha=0.7)
    x_vals = np.linspace(res_centered.min(), res_centered.max(), 200)
    plt.plot(x_vals, stats.norm.pdf(x_vals, 0, sigma_L), 'r-', lw=2)
    plt.xlabel('Residual (dB)')
    plt.ylabel('Probability Density')
    plt.title('Histogram of 1D Centered Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'residuals_histogram.png'), dpi=300)
    plt.show()
    sigma_L=var_res
    sigma_L2=res_centered.std(ddof=1)

    # 5) Link-reliability vs. range using full-grid fit
    Psens_dBm = env.receivers[0].sensitivity
    freq_MHz = rt.frequency / 1e6
    reliabilities = np.linspace(0.001, 0.99, 200)
    ranges_fitted = np.zeros_like(reliabilities)
    ranges_fitted2 = np.zeros_like(reliabilities)
    for i, p in enumerate(reliabilities):
        margin = sigma_L * np.sqrt(2) * erfcinv(2 * (1 - p))
        margin = sigma_L2 * np.sqrt(2) * erfcinv(2 * (1 - p))
        ranges_fitted[i] = compute_max_range_fitted(
            Ptx_dBm, Psens_dBm, margin, rt.G_TX, L0_d0_full, n_full)
        ranges_fitted2[i] = compute_max_range_fitted(
            Ptx_dBm, Psens_dBm, margin, rt.G_TX, L0_d0_line, n_line)
        print(f"  {int(p * 100)}%: M = {margin:.3f} dB")
        print(f"{int(p * 100)}% reliability: fitted range = {ranges_fitted[i]:.3f}")# m, Friis range = {d_fs:.3f} m")


    plt.figure(figsize=(8,5))
    plt.plot(ranges_fitted, reliabilities, 'r-', lw=2)
    plt.plot(ranges_fitted2, reliabilities, 'b-', lw=2)

    plt.xlabel('Range (m)')
    plt.ylabel('Link Reliability')
    plt.title('Reliability vs. Range ')
    plt.grid(True)
    plt.xlim(0,1250)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'reliability_vs_range.png'), dpi=300)
    plt.show()
    sigma_L=res_centered.std(ddof=1)
    # … after computing sigma_L = var_res …
    print(f"\nResidual RMS σ_L = {sigma_L:.2f} dB")


    print(f"\nLo .3f {L0_d0_full}")
    print(f"\n nfull .3f {n_full}")
    # Then your existing loop printing d_fit, d_fs…
    for p in [0.50, 0.95, 0.99]:
        margin = sigma_L * np.sqrt(2) * erfcinv(2 * (1 - p))
        d_fit = compute_max_range_fitted(
            Ptx_dBm, Psens_dBm, margin, rt.G_TX, L0_d0_line, n_line)
        d_fs = compute_max_range_Friis(
            Ptx_dBm, Psens_dBm, margin, n_full, rt.G_TX, freq_MHz)
        print(f"{int(p * 100)}% reliability: fitted range = {d_fit:.3f} m, Margin = {margin:.3f} dB")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()