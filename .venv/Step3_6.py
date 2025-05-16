import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import os
outdir=r'C:/Users/alexb/OneDrive - Université Libre de Bruxelles/MA1-ULBDrive/ELEC-H415/ELEC-H415_V2V/Plot'
os.makedirs(outdir, exist_ok=True)
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from environment import Environment
from raytracing import RayTracing
from position import Position
from receiver import Receiver
from heatmap import create_heatmap
from scipy.special import erfcinv
import multiprocessing

def fit_path_loss(distances, L0_dB):
    x = np.log10(distances)
    y = L0_dB
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    n = slope / 10
    return intercept, n, r_value**2


def compute_max_range_Friis(Ptx_dBm, Psens_dBm, margin_dB, n, G_TX, freq_MHz):
    C = 20*np.log10(freq_MHz) - 147.5 - 20*np.log10(G_TX)
    budget = Ptx_dBm - Psens_dBm - margin_dB
    exp = (budget - C) / (10*n)
    return 10**exp

def compute_max_range_fitted(Ptx_dBm, Psens_dBm, margin_dB,G, L0_d0, n):
    # Available link budget after fading margin
    budget = Ptx_dBm - Psens_dBm - margin_dB +20*np.log10(G)
    exponent = (budget - L0_d0) / (10 * n)
    d_max    = 10**exponent

    return d_max


def main():
    env = Environment()
    env.emitters[0].position = Position(500, 10)
    rt  = RayTracing(env)

    res = 5
    width, height = 1000, 21
    xs = np.arange(0, width, res)
    ys = np.arange(0, height, res)
    X, Y = np.meshgrid(xs, ys)

    P_meas = create_heatmap(env, width, height, res)
    Ptx_dBm = 10 * np.log10(rt.P_TX / 1e-3)

    # 4) Extraction de la transectée au niveau de l'émetteur
    em = env.emitters[0].position
    row_idx = np.argmin(np.abs(ys - em.y))
    d_line = np.arange(0, width, res)
    #d_line   = np.hypot(xs - em.x, np.full_like(xs, em.y) - em.y)
    x_pos = em.x + d_line
    idx   = np.clip(np.round(x_pos / res).astype(int), 0, len(xs)-1)
    L_brut_line = Ptx_dBm - P_meas[row_idx, idx] #dBm
    L0_line = L_brut_line + 20*np.log10(rt.G_TX)
    mask = d_line > 0
    # 5) Fit path-loss sur la transectée 1D
    L0_d0, n, R2 = fit_path_loss(d_line[mask], L0_line[mask])
    print(f"Fit: L0(d0)={L0_d0:.3f} dB, n={n:.4f}, R²={R2:.3f}")

    # 6) Plot Path Loss vs Distance
    d_vals  = np.linspace(0.1, width, 200)
    #d_vals   = np.linspace(d_line[mask].min(), d_line[mask].max(), 200)
    L0_fit = L0_d0 + 10*n*np.log10(d_vals)
    G = rt.G_TX
    L_friis = 20*np.log10(d_vals) + 20*np.log10(rt.frequency) - 147.55 - 20*np.log10(G)

    plt.figure(figsize=(8,5))
    plt.plot(d_line, L0_line, 'o', ms=4, label='Averaged Received Power')
    plt.plot(d_vals, L0_fit, 'r-', lw=2, label=f'Fitted Path Loss {L0_d0:.3f}+10·{n:.4f}log10(d)')
    plt.plot(d_vals, L_friis, 'k--', label='FSPL')
    plt.xlabel('Distance (m)')
    plt.ylabel('Path Loss L0(d) (dB)')
    plt.title('Path Loss vs Distance ')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'pathloss_dist.png'))
    plt.show()

    L0_pred = L0_d0 + 10 * n * np.log10(d_line[mask])
    residuals = L0_line[mask] - L0_pred
    #residuals=Ptx_dBm-P_meas
    residuals_centered = residuals - np.mean(residuals)
    sigma_L = np.std(residuals, ddof=1)
    print(f"Variability σ_L around the model: {sigma_L:.2f} dB")
    stat, p_value = stats.kstest(residuals_centered, 'norm', args=(0, sigma_L))
    print(f"KS test: statistic= {stat:.3f}, p-value= {p_value:.3f}")
    plt.figure(figsize=(8, 5))
    plt.hist(residuals_centered, bins=50, density=True, alpha=0.7)
    x_plot = np.linspace(residuals_centered.min(), residuals_centered.max(), 200)
    plt.plot(x_plot, stats.norm.pdf(x_plot, 0, sigma_L), 'r-', lw=2)
    plt.xlabel('Residual (dB)')
    plt.ylabel('Probability Density')
    plt.title('Histogram of Centered Residuals')
    plt.grid(True)
    plt.show()

    # Step 3.7
    Psens_dBm = env.receivers[0].sensitivity
    freq_MHz = rt.frequency / 1e6
    reliabilities = np.linspace(0.001, 0.99, 200)
    ranges_fitted = np.zeros_like(reliabilities)
    for i, p in enumerate(reliabilities):
        z = stats.norm.ppf(p)
        margin = sigma_L * z
        ranges_fitted[i] = compute_max_range_fitted( Ptx_dBm,Psens_dBm,margin,rt.G_TX,L0_d0,n)
    for p in [0.50, 0.95, 0.99]:
        #z = stats.norm.ppf(p)   #exactly same using other library
        #margin = sigma_L * z
        margin= sigma_L * np.sqrt(2) * erfcinv(2*(1 - p))
        d_fit = compute_max_range_fitted(Ptx_dBm, Psens_dBm, margin,rt.G_TX , L0_d0, n)
        d_fs = compute_max_range_Friis(Ptx_dBm, Psens_dBm, margin, rt.pl_exponent, rt.G_TX , freq_MHz)
        print(f"{int(p * 100)}%: Fade Margin = {margin:.2f}dB → Fitted Range = {d_fit:.3f} m, "
              f"Friis Range = {d_fs:.0f} m")

    # plot only the fitted-model curve
    plt.figure(figsize=(8, 5))
    plt.plot(ranges_fitted, reliabilities, 'r-', lw=2, label='Fitted‐model range')
    plt.xlabel('Maximum Range d (m)')
    plt.ylabel('Reliability P')
    plt.title('Link Reliability vs. Range (Fitted Model)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.0, 1.0)
    plt.xlim(0, ranges_fitted.max() * 1.05)
    plt.savefig(os.path.join(outdir, 'reliability_dist.png'))
    plt.show()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

