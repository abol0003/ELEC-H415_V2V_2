import numpy as np
import matplotlib
# Backend non-interactif
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from environment import Environment
from raytracing import RayTracing
from physics import calculer_coeff_reflexion

# Vitesse de la lumière
c = 299_792_458


def compute_k_factor(rt, emitter, receiver):
    """Calcule le facteur K linéaire (Rician) pour un emplacement donné du récepteur."""
    # Ordre 0 (LOS)

    P0,_=rt.direct_propagation(emitter,receiver)
    P1,_=rt.reflex_and_power(emitter,receiver)
    P2,_=rt.double_reflex_and_power(emitter,receiver)
    P3,_=rt.triple_reflex_and_power(emitter,receiver)

    # Retour du facteur K linéaire
    return P0 / (P1 + P2 + P3)


def rice_pdf(r, K, omega=1.0):
    """PDF de la distribution de Rice pour amplitude normalisée r, paramètre K et puissance moyenne omega."""
    # Paramètres s et sigma selon définition de Rice
    s = np.sqrt(K * omega / (K + 1)) #power in direct path
    sigma = np.sqrt(omega / (2 * (K + 1))) # averaging non LOS model source wikipedia
    # Utilisation de numpy.i0 pour la fonction de Bessel I0
    return (r / sigma**2) * np.exp(-(r**2 + s**2) / (2 * sigma**2)) * np.i0(r * s / sigma**2)


def main():
    # Initialisation du ray tracing
    env = Environment()
    rt = RayTracing(env)
    emitter = env.emitters[0]
    receiver = env.receivers[0]

    # ----------------------
    # Plot 1: Facteur K vs distance
    # ----------------------
    distances = np.linspace(5, 200, 100)
    K_vals = []
    for d in distances:
        receiver.position.x = emitter.position.x + d
        K_vals.append(compute_k_factor(rt, emitter, receiver))

    K_vals = np.array(K_vals)
    K_dB = 10 * np.log10(K_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(distances, K_vals, label='K (linéaire)')
    plt.plot(distances, K_dB, '--', label='K (dB)')
    plt.xlabel('Distance émetteur–récepteur (m)')
    plt.ylabel('Facteur K Rician')
    plt.title('Variation du facteur K Rician avec la distance')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    #plt.savefig('rician_K_vs_distance.png', dpi=300)
    print("Figure enregistrée : rician_K_vs_distance.png")
    plt.show()

    # ----------------------
    # Plot 2: PDF de Rice vs |h|
    # ----------------------
    h = np.linspace(0, 8, 400)
    Ks = [0, 2, 8, 32]

    plt.figure(figsize=(8, 5))
    for K in Ks:
        pdf = rice_pdf(h, K, omega=1.0)
        plt.plot(h, pdf, label=f'K = {K}')

    plt.xlabel('|h|')
    plt.ylabel('PDF')
    plt.title('PDF de la distribution de Rice pour différents K')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    #plt.savefig('rice_pdf_vs_h.png', dpi=300)
    print("Figure enregistrée : rice_pdf_vs_h.png")
    plt.show()


if __name__ == '__main__':
    main()