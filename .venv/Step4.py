
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
outdir=r'C:/Users/alexb/OneDrive - Université Libre de Bruxelles/MA1-ULBDrive/ELEC-H415/ELEC-H415_V2V/Plot'
os.makedirs(outdir, exist_ok=True)
from environment import Environment
from raytracing import RayTracing


def step4_1_wideband_LOS(env, rt, BRF=100e6, Nt=1000, Nf=1000):
    tx   = env.emitters[0]
    rx   = env.receivers[0]
    c    = 299_792_458                    # speed of light (m/s)
    d1   = rt.calc_distance(tx.position, rx.position)
    tau1 = d1 / c                         # propagation delay (s)
    fc   = rt.frequency                   # carrier frequency (Hz)

    # 1) Sampled impulse response h(τ)
    t_max = 2 * tau1
    t     = np.linspace(0, t_max, Nt)
    h     = np.zeros(Nt, dtype=complex)
    idx   = np.argmin(np.abs(t - tau1))
    # include amplitude and carrier phase at τ1
    h[idx] = np.exp(-1j * 2*np.pi * fc * tau1) / d1

    # Plot |h(τ)|
    plt.figure(figsize=(8,4))
    plt.plot(t * 1e9, np.abs(h),'-',markersize=6,linewidth=2)
    plt.xlabel('Delay τ (ns)')
    plt.ylabel('|h(τ)|')
    plt.title('LOS Impulse Response Magnitude')
    plt.xlim(0, t_max*1e9)  # limit x-axis from 0 to 2*tau1 in ns
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/impulse_magnitude.png")

    plt.show()

    # Plot ∠h(τ)
    plt.figure(figsize=(8,4))
    plt.plot(t * 1e9, np.degrees(np.angle(h)),'-',markersize=6,linewidth=2)
    plt.xlabel('Delay τ (ns)')
    plt.ylabel('Phase(h(τ)) (°)')
    plt.title('LOS Impulse Response Phase')
    plt.xlim(0, t_max*1e9)
    plt.grid(True)
    plt.tight_layout()
    #plt.show()

    # 2) Analytical frequency response H(f) = (1/d1)·exp(-j2πfτ1)
    f = np.linspace(fc - BRF/2, fc + BRF/2, Nf)
   # H = (1.0/d1) * np.exp(-1j * 2*np.pi * f * tau1)
    H = np.fft.fftshift(np.fft.fft(h))
    # Plot |H(f)|
    plt.figure(figsize=(8,4))
    plt.plot((f-fc)*1e-6, np.abs(H), 'C0-')
    plt.xlabel('Frequency Offset (MHz)')
    plt.ylabel('|H(f)|')
    plt.title('LOS Frequency Response Magnitude')
    plt.xlim(-BRF/2*1e-6-5, BRF/2*1e-6+5)  # set x-axis ±50 MHz
    plt.xticks(np.arange(-50, 51, 25))  # ticks at -50, -25, 0, 25, 50
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/frequency_magnitude.png")

    plt.show()
def step4_2_TDL_wideband(env, rt, BRF=100e6):
    """
    Step 4.2: Tapped‐Delay‐Line for the LOS ray only, with finite bandwidth:
      h_TDL(τ) = Σ_{l=0}^L α₁ · sinc( BRF·(τ₁ - l·Δτ) ) · δ( τ - l·Δτ )
    where Δτ = 1/BRF,  α₁ = exp(-j2π f_c τ₁)/d₁, and τ₁=d₁/c.
    """
    # 1) TX/RX geometry
    tx   = env.emitters[0]
    rx   = env.receivers[0]
    c    = 299_792_458
    d1   = rt.calc_distance(tx.position, rx.position)
    tau1 = d1 / c
    fc   = rt.frequency

    # 2) Time‐bin Δτ and tap indices l
    dtau = 1/BRF
    L    = int(np.ceil(2*tau1/dtau))
    l    = np.arange(0, L+1)
    t_l  = l * dtau

    # 3) Complex amplitude α₁ for the LOS ray
    alpha1 = np.exp(-1j * 2*np.pi * fc * tau1) / d1
    # 4) Compute tapped amplitudes h_l with sinc envelope
    h_l = alpha1 * np.sinc(BRF * (tau1 - t_l))

    # 5) Plot |h_TDL(τ)| vs delay
    plt.figure(figsize=(8,4))
    plt.plot(t_l * 1e9, np.abs(h_l),'-',markersize=6,linewidth=2)
    plt.xlabel('Delay τ (ns)')
    plt.ylabel('|$h_{TDL}(τ)$|')
    plt.title('$h_{TDL}(τ)$ Impulse Response')
    plt.xlim(0, t_l.max()*1e9)
    plt.grid(True)
    plt.tight_layout()

    # 6) Save to disk
    if 'outdir' in globals():
        os.makedirs(outdir, exist_ok=True)
        fname = os.path.join(outdir, 'tdl_wideband.png')
        plt.savefig(fname, dpi=300)
        print(f"Saved wideband TDL plot to {fname}")
    plt.show()

if __name__ == '__main__':
    env = Environment()
    rt  = RayTracing(env)
    step4_1_wideband_LOS(env, rt)
    step4_2_TDL_wideband(env, rt, BRF=100e6)
