import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from environment import Environment
from raytracing import RayTracing
from position import Position
from receiver import Receiver
from heatmap import create_heatmap

def calculate_power(x, y, env, rt, use_pl=False):
    """
    Calculate the received signal power (dBm) at position (x, y) using the ray tracer.
    If use_pl=True, enable the internal path-loss model.
    """
    dummy_receiver = Receiver(Position(x, y), sensitivity=-70)
    env.receivers = [dummy_receiver]
    rt.use_path_loss = use_pl
    rt.ray_tracer()
    rt.use_path_loss = False
    received = dummy_receiver.received_power_dBm
    return received if received >= -90 else -90


def sigmaL_okumura_hata(f_MHz):
    """
    Empirical equation (3.56) for shadow fading variability σ_L as a function of frequency.
    """
    return 0.65 * np.log10(f_MHz)**2 - 1.3 * np.log10(f_MHz) + 5.2


def main():
    # Initialize environment and ray tracer
    env = Environment()
    rt = RayTracing(env)

    # Resolution and grid setup (meters)
    res = 5
    width, height = 1000, 22
    xs = np.arange(0, width, res)
    ys = np.arange(0, height, res)
    X, Y = np.meshgrid(xs, ys)

    # Heatmaps of received power
    P_measured = create_heatmap(env, width=width, height=height, resolution=res, with_pl=False)
    P_modeled  = create_heatmap(env, width=width, height=height, resolution=res, with_pl=True)

    # Compute residuals between measured and modeled
    residues = P_measured - P_modeled

    # Estimate σ_L from residuals (sample standard deviation)
    sigma_L_measured = np.std(residues, ddof=1)
    print(f"Estimated σ_L (measured): {sigma_L_measured:.2f} dB")

    # Perform Kolmogorov-Smirnov test for normality of residuals
    mu, std = np.mean(residues), np.std(residues, ddof=1)
    stat, p_value = stats.kstest(residues.ravel(), 'norm', args=(mu, std))
    stat = float(stat)
    p_value = float(p_value)
    print(f"KS test: statistic={stat:.3f}, p-value={p_value:.3f}")

    # Plot histogram of residuals with fitted normal density
    plt.figure(figsize=(8, 5))
    plt.hist(residues.ravel(), bins=20, density=True, alpha=0.7, label='Residuals')
    x_plot = np.linspace(residues.min(), residues.max(), 200)
    plt.plot(x_plot, stats.norm.pdf(x_plot, mu, std), linewidth=2,
             label=f'N({mu:.2f}, {std:.2f}\u00b2)')
    plt.xlabel('ΔP (dB)')
    plt.ylabel('Density')
    plt.title('Histogram of Residuals and Normal PDF')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Compute and bin distances from emitter, then compute σ_L per distance bin
    emitter_pos = env.emitters[0].position
    distances = np.sqrt((X - emitter_pos.x)**2 + (Y - emitter_pos.y)**2).ravel()
    resid_flat = residues.ravel()

    # Define radial bins (e.g., 20 bins)
    num_bins = 20
    bins = np.linspace(distances.min(), distances.max(), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    sigma_per_bin = []

    # Compute σ_L for each bin
    for i in range(num_bins):
        mask = (distances >= bins[i]) & (distances < bins[i+1])
        if np.any(mask):
            sigma_per_bin.append(std if not np.any(mask) else np.std(resid_flat[mask], ddof=1))
        else:
            sigma_per_bin.append(np.nan)

    # Plot σ_L vs distance bins
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, sigma_per_bin, marker='o')
    plt.xlabel('Distance from Emitter (m)')
    plt.ylabel('σ_L (dB)')
    plt.title('σ_L vs. Distance Bins')
    plt.grid(True)
    plt.show()

    # Theoretical σ_L vs frequency according to Okumura-Hata equation (3.56)
    f_MHz = np.linspace(1, 20000, 200)  # 1 MHz to 20 GHz
    f_meas = 5900                       # 5.9 GHz
    sigma_theoretical = sigmaL_okumura_hata(f_MHz)

    plt.figure(figsize=(8, 5))
    plt.plot(f_MHz, sigma_theoretical, linewidth=2, label='Eq. 3.56')
    plt.scatter(f_MHz, sigma_theoretical, s=10)
    plt.scatter([f_meas], [sigma_L_measured], s=50, marker='x',
                label=f'Measured σ_L @ {f_meas/1000:.1f} GHz')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('σ_L (dB)')
    plt.title('σ_L Variability vs Frequency (Figure 3.13)')
    plt.xlim(f_MHz.min(), f_MHz.max())
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    main()
