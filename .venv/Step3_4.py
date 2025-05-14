import numpy as np
import matplotlib
# Use non-interactive backend
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from environment import Environment
from raytracing import RayTracing
from physics import calculer_coeff_reflexion

# Speed of light (m/s)
c = 299792458

def compute_k_factor(rt, emitter, receiver):
    """
    Compute the linear Rician K-factor for a given receiver location.
    """
    P0, _ = rt.direct_propagation(emitter, receiver)
    P1, _ = rt.reflex_and_power(emitter, receiver)
    P2, _ = rt.double_reflex_and_power(emitter, receiver)
    P3, _ = rt.triple_reflex_and_power(emitter, receiver)
    return P0**2 / (P1[0]**2+P1[1]**2 + P2[0]**2 +P2[1]**2 + P3[0]**2+ P3[1]**2)

def rice_pdf(r, K, omega=1.0):
    """
    Compute the Rice distribution PDF for normalized amplitude r,
    Rician factor K, and mean power omega.
    """
    # Parameters s and sigma
    s = np.sqrt(K * omega / (K + 1))  # LOS component amplitude
    sigma = np.sqrt(omega / (2 * (K + 1)))  # NLOS component scale
    # Rice PDF formula using modified Bessel I0
    return (r / sigma**2) * np.exp(-(r**2 + s**2) / (2 * sigma**2)) * np.i0(r * s / sigma**2)

def main():
    # Initialize environment and ray tracer
    env = Environment()
    rt = RayTracing(env)
    emitter = env.emitters[0]
    receiver = env.receivers[0]

    # ----------------------
    # Plot 1: K-Factor vs Distance
    # ----------------------
    distances = np.linspace(0, 1000, 1000)
    K_vals = []
    for d in distances:
        rt.rice=True
        receiver.position.x = emitter.position.x + d
        K_vals.append(compute_k_factor(rt, emitter, receiver))

    K_vals = np.array(K_vals)
    K_dB = 10 * np.log10(K_vals)

    plt.figure(figsize=(8, 5))
    #plt.plot(distances, K_vals, label='K (linear)')
    plt.plot(distances, K_dB, '-', label='K (dB)')
    plt.xlabel('Transmitter–Receiver Distance (m)')
    plt.ylabel('Rician K-Factor')
    plt.title('Rician K-Factor Variation with Distance')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig('rician_K_vs_distance.png', dpi=300)
    print("Figure saved: rician_K_vs_distance.png")
    plt.show()

    # ----------------------
    # Plot 2: Rice PDF vs |h|
    # ----------------------
    h = np.linspace(0, 8, 400)
    Ks = [0, 2, 8, 32]

    plt.figure(figsize=(8, 5))
    for K in Ks:
        pdf = rice_pdf(h, K, omega=1.0)
        plt.plot(h, pdf, label=f'K = {K}')

    plt.xlabel('Amplitude |h|')
    plt.ylabel('PDF')
    plt.title('Rice Distribution PDF for Various K Values')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig('rice_pdf_vs_h.png', dpi=300)
    print("Figure saved: rice_pdf_vs_h.png")
    plt.show()

if __name__ == '__main__':
    main()
