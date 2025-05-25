#!/usr/bin/env python3
"""
Step6: Full Space-Time-Frequency Channel Analysis (Chapters 4 & 8)

Performs:
  1) Power Delay Profile, Delay Spread, Coherence Bandwidth
  2) Empirical Doppler Spectrum & Coherence Time
  3) Delay–Doppler Scattering Function
  4) Received Spectrum Impulses around Carrier
  5) Temporal Autocorrelation
  6) Frequency Correlation
"""
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.special import j0

from environment import Environment
from raytracing import RayTracing
from Step3_3 import gather_mpcs

OUT_DIR = r'C:/Users/alexb/OneDrive - Université Libre de Bruxelles/MA1-ULBDrive/ELEC-H415/ELEC-H415_V2V/Plot'
os.makedirs(OUT_DIR, exist_ok=True)
C = 299_792_458


def analyze_power_delay_profile(mpcs, B):
    """
    mpcs : liste de dicts avec 'delay' (s) et 'amplitude' (complex ou float)
    B    : bande passante (Hz) => Δτ = 1/B
    """
    # pas d'échantillonnage en temps
    delta_tau = 1 / B
    # retards max et nombre de taps
    tau_max = max(p['delay'] for p in mpcs)
    L = int(np.ceil(tau_max / delta_tau)) + 1

    # initialisation des taps
    h_l = np.zeros(L, dtype=complex)
    for p in mpcs:
        l = int(np.round(p['delay'] / delta_tau))
        h_l[l] += p.get('amplitude', 1.0)

    # Power Delay Profile
    P = np.abs(h_l)**2
    P /= P.sum()

    # vecteur des retards
    taus = np.arange(L) * delta_tau

    # affichage
    plt.figure(figsize=(8,4))
    m, s, b = plt.stem(taus*1e9, P)
    plt.setp(s, 'linewidth', 1)
    plt.setp(b, 'visible', False)
    plt.xlabel('Delay τ (ns)')
    plt.ylabel('Normalized power')
    plt.title('Power Delay Profile (échantillonné)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'pdp.png'), dpi=300)
    plt.show()

    # statistiques
    tau_mean = (taus * P).sum()
    tau_rms  = np.sqrt(((taus - tau_mean)**2 * P).sum())
    print(f"Mean delay: {tau_mean*1e9:.2f} ns, RMS spread: {tau_rms*1e9:.2f} ns")
    print(f"Coherence bandwidth ≈ {1/(2*np.pi*tau_rms)/1e6:.2f} MHz")


def analyze_doppler(mpcs, fc, v):
    lam = C / fc
    fD = np.array([
        (v/lam) * np.cos(np.deg2rad(a))
        for p in mpcs for a in p.get('angles', [0.0])
    ])

    # empirical
    plt.figure(figsize=(8,4))
    h, e = np.histogram(fD, bins=100, density=True)
    c = (e[:-1] + e[1:]) / 2
    plt.bar(c, h, width=e[1]-e[0], alpha=0.6)
    plt.xlabel('Doppler shift f_D (Hz)')
    plt.ylabel('Density')
    plt.title('Empirical Doppler Spectrum')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'doppler_emp.png'), dpi=300)
    plt.show()

    # coherence time
    fD_max = v / lam
    Tc = lam / (2*v)
    print(f"Coherence time ≈ {Tc*1e3:.3f} ms")
    return fD, fD_max


def plot_scattering(mpcs, fc, v):
    lam = C / fc
    taus = np.array([p['delay'] for p in mpcs])
    fD = np.array([(v/lam)*np.cos(np.deg2rad(a))
                   for p in mpcs for a in p.get('angles',[0.0])])

    tb = np.linspace(0, taus.max()*1.1, 200)
    fb = np.linspace(-v/lam, v/lam, 200)
    S, _, _ = np.histogram2d(
        np.repeat(taus, [len(p.get('angles',[0.0])) for p in mpcs]),
        fD, bins=[tb, fb], density=True)

    plt.figure(figsize=(6,5))
    plt.imshow(S.T, origin='lower', aspect='auto',
               extent=[0, tb.max()*1e9, -fb.max(), fb.max()])
    plt.xlabel('Delay τ (ns)')
    plt.ylabel('Doppler f_D (Hz)')
    plt.title('Scattering Function S(τ,f_D)')
    plt.colorbar(label='Density')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'scattering.png'), dpi=300)
    plt.show()


def plot_received_spectrum(env, rt, fc, v):
    lam = C/fc
    mpcs = gather_mpcs(rt, env.emitters[0], env.receivers[0])

    freqs, amps = [], []
    for p in mpcs:
        for a in p.get('angles',[0.0]):
            fD = (v/lam)*np.cos(np.deg2rad(a))
            freqs.append(fc + fD)
            amps.append(p.get('amplitude',1.0))
    freqs = np.array(freqs)
    amps = np.array(amps) / np.max(amps)

    plt.figure(figsize=(8,4))
    m, s, b = plt.stem(freqs*1e-6, amps)
    plt.setp(s, 'linewidth',1); plt.setp(b,'visible',False)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Normalized amplitude')
    plt.title(f'Received Spectrum around {fc*1e-6:.2f} MHz')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'received.png'), dpi=300)
    plt.show()


def temporal_autocorr(fD, fD_max):
    taus = np.arange(0, 2/fD_max, 1e-4)
    Rt = np.abs(np.mean(np.exp(1j*2*np.pi*np.outer(fD, taus)), axis=0))
    Rth = j0(2*np.pi*fD_max*taus)

    plt.figure(figsize=(8,4))
    plt.plot(taus*1e3, Rt, label='Result')
    plt.plot(taus * 1e3, Rth, '-',
             label=r'$J_0(2\pi f_{D,\max}\,\Delta t)$')
    plt.xlabel('Lag Δt (ms)')
    plt.ylabel('|R_t(Δt)|')
    plt.title('Temporal Autocorrelation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'temporal_autocorr.png'), dpi=300)
    plt.show()


def frequency_correlation(H, f):
    Rf = (np.fft.ifft(np.abs(H)**2))
    df = f[1] - f[0]
    lags = np.arange(0, len(f)) * df

    plt.figure(figsize=(8,4))
    plt.plot(lags*1e-6, np.abs(Rf))
    plt.xlabel('Frequency lag Δf (MHz)')
    plt.ylabel('Normalized R_f')
    plt.xlim(-20, 20)
    plt.title('Frequency Correlation')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'frequency_corr.png'), dpi=300)
    plt.show()


if __name__ == '__main__':
    env = Environment()
    rt  = RayTracing(env)

    # Step 1 PDP
    mpcs = gather_mpcs(rt, env.emitters[0], env.receivers[0])
    analyze_power_delay_profile(mpcs,100e6)

    # Step 2 Doppler
    fc = rt.frequency
    v = 50/3.6
    fD, fD_max = analyze_doppler(mpcs, fc, v)

    # Step 3 Scattering
    plot_scattering(mpcs, fc, v)

    # Step 4 Received spectrum
    plot_received_spectrum(env, rt, fc, v)

    # Step 5 Temporal autocorr
    temporal_autocorr(fD, fD_max)

    # Step 6 Frequency correlation
    N = 1024
    f = np.linspace(fc-50e6, fc+50e6, N)
    H = np.zeros_like(f, dtype=complex)
    for p in mpcs:
        for a in p.get('angles',[0.0]):
            H += p.get('amplitude',1.0)*np.exp(-1j*2*np.pi*(f)*p['delay'])
    frequency_correlation(H, f)

