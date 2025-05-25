import numpy as np
import matplotlib
import os
outdir=r'C:/Users/alexb/OneDrive - Université Libre de Bruxelles/MA1-ULBDrive/ELEC-H415/ELEC-H415_V2V/Plot'
os.makedirs(outdir, exist_ok=True)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from environment import Environment
from raytracing import RayTracing
from physics import calculer_coeff_reflexion

# Speed of light (m/s)
c = 299792458

def compute_k_factor(rt, emitter, receiver,inter):
    """
    Compute the linear Rician K-factor for a given receiver location.
    """
    P0, v0 = rt.direct_propagation(emitter, receiver)
    P1, v1 = rt.reflex_and_power(emitter, receiver)
    P2, v2 = rt.double_reflex_and_power(emitter, receiver)
    P3, v3 = rt.triple_reflex_and_power(emitter, receiver)
    Power0 = (np.abs( v0) ** 2) / (8 * rt.Ra)
    Power = (np.abs( v1 + v2 + v3) ** 2) / (8 * rt.Ra)
    #-return Power0**2/ Power**2
    return (Power0 ** 2 / Power ** 2) if inter else (
            P0 ** 2 / (
            P1[0] ** 2 + P1[1] ** 2 +
            P2[0] ** 2 + P2[1] ** 2 +
            P3[0] ** 2 + P3[1] ** 2
    )
    )
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
    distances = np.linspace(1, 1000, 1000)
    K_vals = []
    K_vals_w = []
    for d in distances:
        rt.rice=True
        receiver.position.x = emitter.position.x + d
        K_vals.append(compute_k_factor(rt, emitter, receiver,False))
        #K_vals_w.append(compute_k_factor(rt, emitter, receiver,True))


   # K_vals = np.array(K_vals)
    K_dB = 10 * np.log10(K_vals)
    #K_dB[0]=40
    #K_vals = np.array(K_vals_w)
    #K_dB_2 = 10 * np.log10(K_vals)

    plt.figure(figsize=(8, 5))
   # plt.plot(distances, K_vals, label='K (linear)')
    plt.plot(distances, K_dB, '-', label='K (dB)')
    #plt.plot(distances, K_dB_2, '-', label='K (dB) Real Values')

    plt.xlabel('Transmitter–Receiver Distance (m)')
    plt.ylabel('Rician K-Factor (dB)')
    plt.title('Rice Factor Variation with Distance')
    plt.legend()
    plt.ylim(-10,50)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rician_K_vs_distance.png'))
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
    plt.savefig(os.path.join(outdir, 'rice_pdf_vs_h.png'))
    plt.show()

if __name__ == '__main__':
    main()
