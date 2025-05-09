"""
step3_6.py
Calcul de la variabilité σ_L des résidus autour du modèle path-loss implémenté dans RayTracing
Ajout : calcul théorique de σ_L selon l'équation 3.56 et tracé selon Figure 3.13
"""
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


def calculate_power(x, y, env, rt, use_pl=False):
    """
    Calcule la puissance reçue (dBm) à la position (x, y) en utilisant le ray-tracer.
    Si use_pl=True, active le path-loss interne.
    """
    """
    Calcule la puissance de signal reçue en dBm à un point spécifique.
    Utilise un récepteur fictif pour mesurer la puissance.
    """
    dummy_receiver = Receiver(Position(x, y), sensitivity=-70)
    env.receivers = [dummy_receiver]
    rt.use_path_loss = use_pl
    rt.ray_tracer()
    rt.use_path_loss = False
    return dummy_receiver.received_power_dBm if dummy_receiver.received_power_dBm >= -90 else -90


def sigmaL_okumura_hata(f_MHz):
    """
    Équation empirique 3.56 :
    f_MHz : fréquence en MHz
    """
    return 0.65 * (np.log10(f_MHz))**2 - 1.3 * np.log10(f_MHz) + 5.2


def main():
    # Initialisation de l'environnement et du ray-tracer
    env0 = Environment()
    rt = RayTracing(env0)

    # Récupération de l'émetteur principal pour position de référence
    emitter = env0.emitters[0]
    x0, y0 = emitter.position.x, emitter.position.y

    # Génération des positions sur la ligne tous les 5 m
    max_distance = 1000+ 1e-9 # en mètrese-
    distances = np.arange(0, max_distance, 5)
    positions = [(x0 + d, y0) for d in distances]

    # Calcul multithreaded de la puissance sans et avec path-loss
    func_no_pl = partial(calculate_power, env=env0, rt=rt, use_pl=False)
    func_pl = partial(calculate_power, env=env0, rt=rt, use_pl=True)
    with ThreadPoolExecutor() as executor:
        P_mesuree = np.array(list(executor.map(lambda p: func_no_pl(p[0], p[1]), positions)))
    with ThreadPoolExecutor() as executor:
        P_modele = np.array(list(executor.map(lambda p: func_pl(p[0], p[1]), positions)))

    # Calcul des résidus
    residues = P_mesuree - P_modele

    # Estimation de σ_L (mesuré)
    sigma_L_mesure = np.std(residues, ddof=1)
    print(f"σ_L estimé (mesuré) : {sigma_L_mesure:.2f} dB")

    # Test de normalité (Kolmogorov-Smirnov)
    mu, std = np.mean(residues), np.std(residues, ddof=1)
    stat, p_value = stats.kstest(residues, 'norm', args=(mu, std))
    print(f"KS test : stat={stat:.3f}, p-value={p_value:.3f}")

    # Tracé : histogramme des résidus + densité normale
    plt.figure(figsize=(8, 5))
    plt.hist(residues, bins=20, density=True, alpha=0.7, label='Résidus')
    x = np.linspace(residues.min(), residues.max(), 200)
    plt.plot(x, stats.norm.pdf(x, mu, std), linewidth=2, label=f'N({mu:.2f},{std:.2f}²)')
    plt.xlabel('ΔP (dB)')
    plt.ylabel('Densité')
    plt.title('Histogramme des résidus et densité normale')
    plt.legend()
    plt.grid(True)
    plt.show()

        # Tracé : évolution de σ_L selon la distance
    window_size = 5
    centers = []
    sigmas = []
    for i in range(len(residues) - window_size + 1):
        centers.append(np.mean(distances[i:i+window_size]))
        sigmas.append(np.std(residues[i:i+window_size], ddof=1))
    plt.figure(figsize=(8, 5))
    plt.plot(centers, sigmas, marker='o')
    plt.xlabel('Distance depuis émetteur (m)')
    plt.ylabel('σ_L (dB)')
    plt.title('Évolution de σ_L selon la distance')
    plt.grid(True)
    plt.show()

    #     # --- Heatmaps de la puissance reçue avec et sans path-loss ---
    # # Définir la zone de calcul (à ajuster selon l'environnement)
    # width, height = 1000, 20  # en mètres
    # res = 5  # résolution en m
    # xs = np.arange(0, width, res)
    # ys = np.arange(0, height, res)
    # Xg, Yg = np.meshgrid(xs, ys)
    # grid_positions = [(x, y) for x, y in zip(Xg.ravel(), Yg.ravel())]
    #
    # # Calcul multithreaded puissance reçue
    # func_no_pl_grid = partial(calculate_power, env=env0, rt=rt, use_pl=False)
    # func_pl_grid = partial(calculate_power, env=env0, rt=rt, use_pl=True)
    # with ThreadPoolExecutor() as executor:
    #     P_no_pl = np.array(list(executor.map(lambda p: func_no_pl_grid(p[0], p[1]), grid_positions)))
    # with ThreadPoolExecutor() as executor:
    #     P_pl = np.array(list(executor.map(lambda p: func_pl_grid(p[0], p[1]), grid_positions)))
    #
    # P_no_pl_grid = P_no_pl.reshape(ys.size, xs.size)
    # P_pl_grid = P_pl.reshape(ys.size, xs.size)
    #
    # # Heatmap sans path loss
    # fig, ax = plt.subplots(figsize=(12, 4))
    # pcm = ax.pcolormesh(Xg, Yg, P_no_pl_grid, shading='auto', cmap='viridis')
    # plt.colorbar(pcm, ax=ax, label='Puissance reçue (dBm)')
    # ax.set(title='Heatmap puissance reçue sans path-loss', xlabel='X (m)', ylabel='Y (m)')
    # ax.invert_yaxis()
    # plt.show()
    #
    # # Heatmap avec path loss
    # fig, ax = plt.subplots(figsize=(12, 4))
    # pcm = ax.pcolormesh(Xg, Yg, P_pl_grid, shading='auto', cmap='viridis')
    # plt.colorbar(pcm, ax=ax, label='Puissance reçue (dBm)')
    # ax.set(title='Heatmap puissance reçue avec path-loss', xlabel='X (m)', ylabel='Y (m)')
    # ax.invert_yaxis()
    # plt.show()

        # --- Calcul (théorique) de σ_L vs fréquence selon eq.3.56 et tracé Figure 3.13 ---
    # Fréquences théoriques et fréquence mesurée
    f_MHz = np.linspace(1, 20000, 200)  # 100 MHz à 6 GHz
    f_meas = 5900  # 5.9 GHz
    sigma_theorique = sigmaL_okumura_hata(f_MHz)
    sigma_theorique = sigmaL_okumura_hata(f_MHz)
    plt.figure(figsize=(8, 5))
    plt.plot(f_MHz, sigma_theorique, linewidth=2, label='Eq. 3.56')
    plt.scatter(f_MHz, sigma_theorique, s=10)
        # Comparaison avec le modèle mesuré
    # Point mesuré spécifique à 5.9 GHz
    plt.scatter([f_meas], [sigma_L_mesure], color='black', s=50, marker='x', label=f'σ_L mesuré @ {f_meas/1000:.1f} GHz')
    plt.xlabel('Fréquence (MHz)')
    plt.ylabel('σ_L (dB)')
    plt.title('Figure 3.13 – Variabilité σ_L selon la fréquence')
    plt.xlim(f_MHz.min(), f_MHz.max())
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
