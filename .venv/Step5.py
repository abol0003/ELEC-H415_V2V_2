#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from environment import Environment
from raytracing import RayTracing
import physics  # for calculer_coeff_reflexion

# Output directory
outdir = r'C:/Users/alexb/OneDrive - Université Libre de Bruxelles/MA1-ULBDrive/ELEC-H415/ELEC-H415_V2V/Plot'
os.makedirs(outdir, exist_ok=True)

def step5_full_wideband(env, rt, BRF=100e6):
    tx   = env.emitters[0]
    rx   = env.receivers[0]
    c    = 299_792_458
    fc   = rt.frequency
    lamb=c/fc
    taps = []
    d0   = rt.calc_distance(tx.position, rx.position)
    tau0 = d0 / c
    a0, _, _ = rt.compute_electrical_field_and_power(1.0, d0)
    taps.append((tau0, a0/d0))

    # Single reflections
    for obs in env.obstacles:
        img1 = rt.compute_image_position(obs, tx.position)
        if obs.check_intersection(img1, rx.position):
            imp1 = obs.impact_point(img1, rx.position)
            if imp1 is not None:
                coeff1   = physics.calculer_coeff_reflexion(obs, tx.position, imp1)
                d1   = rt.calc_distance(img1, rx.position)
                tau1 = d1 / c
                a1, _, _ = rt.compute_electrical_field_and_power(coeff1, d1)
                taps.append((tau1, a1/d1))

    # Double reflections
    for obs1 in env.obstacles:
        img1 = rt.compute_image_position(obs1, tx.position)
        for obs2 in env.obstacles:
            if obs2 is obs1: continue
            img2 = rt.compute_image_position(obs2, img1)
            if obs2.check_intersection(img2, rx.position):
                imp2 = obs2.impact_point(img2, rx.position)
                if imp2 and obs1.check_intersection(img1, imp2):
                    imp1 = obs1.impact_point(img1, imp2)
                    if imp1 is not None:
                        coeff1 = physics.calculer_coeff_reflexion(obs1, tx.position, imp1)
                        coeff2 = physics.calculer_coeff_reflexion(obs2, img1, imp2)
                        coeff = coeff1 * coeff2
                        d2   = rt.calc_distance(img2, rx.position)
                        tau2 = d2 / c
                        a2, _, _ = rt.compute_electrical_field_and_power(coeff, d2)
                        taps.append((tau2, a2/d2))

    # Triple reflections
    for obs1 in env.obstacles:
        img1 = rt.compute_image_position(obs1, tx.position)
        for obs2 in env.obstacles:
            if obs2 is obs1: continue
            img2 = rt.compute_image_position(obs2, img1)
            for obs3 in env.obstacles:
                if obs3 is obs2: continue
                img3 = rt.compute_image_position(obs3, img2)
                ok3 = obs3.check_intersection(img3, rx.position)
                ok2 = obs2.check_intersection(img2, obs3.impact_point(img3, rx.position) or rx.position)
                ok1 = obs1.check_intersection(img1, obs2.impact_point(img2, obs3.impact_point(img3, rx.position)) or rx.position)
                if ok1 and ok2 and ok3:
                    imp3 = obs3.impact_point(img3, rx.position)
                    imp2 = obs2.impact_point(img2, imp3)
                    imp1 = obs1.impact_point(img1, imp2)
                    if None not in (imp1, imp2, imp3):
                        coeff1 = physics.calculer_coeff_reflexion(obs1, tx.position, imp1)
                        coeff2 = physics.calculer_coeff_reflexion(obs2, img1, imp2)
                        coeff3 = physics.calculer_coeff_reflexion(obs3, img2, imp3)
                        coeff = coeff1 * coeff2 * coeff3
                        d3   = rt.calc_distance(img3, rx.position)
                        tau3 = d3 / c
                        a3, _, _ = rt.compute_electrical_field_and_power(coeff, d3)
                        taps.append((tau3, a3/d3))

    # --- 2) Sampled impulse response h(τ) on [0,2·max τ_i] ---
    max_tau = max(t for t,_ in taps)
    t_max   = 2*max_tau
    Nt = 1000
    t       = np.linspace(0, t_max, Nt)
    h       = np.zeros(Nt, dtype=complex)

    for tau_i, a_i in taps:
        idx = np.argmin(np.abs(t - tau_i))
        h[idx] += a_i/a0  # to put a0 to 1 as previous case then we calculate a_I/a0 to see attenuation du to reflection

    # --- 3) Plot |h(τ)| ---
    plt.figure(figsize=(8,4))
    plt.plot(t*1e9, np.abs((lamb*h/(3*np.pi**2))), '-', markersize=4, linewidth=1.5)
    plt.xlabel('Delay τ (ns)')
    plt.ylabel('|h(τ)|')
    plt.title('Channel Physical Impulse Response')
    plt.xlim(0, t_max*1e9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'impulse_full_magnitude2.png'), dpi=300)
    plt.show()
    h=lamb*h/(3*np.pi**2)
    # --- 4) Plot ∠h(τ) ---
    plt.figure(figsize=(8,4))
    plt.plot(t*1e9, np.degrees(np.angle(h)), '-o', markersize=4, linewidth=1.5)
    plt.xlabel('Delay τ (ns)')
    plt.ylabel('Phase h(τ) (°)')
    plt.title('Channel Impulse Response Phase')
    plt.xlim(0, t_max*1e9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'impulse_full_phase2.png'), dpi=300)
    #plt.show()

    # --- 5) Analytical frequency response H(f) ---
    f = np.linspace(fc - BRF/2, fc + BRF/2, 1000)
    H = np.zeros_like(f, dtype=complex)
    # for tau_i, a_i in taps:
    #     H += a_i * np.exp(-1j*2*np.pi*f*tau_i)
    H = np.fft.fftshift(np.fft.fft(h))
    H2=np.fft.fft(h)
    plt.figure(figsize=(8,4))
    #plt.plot((f-fc)*1e-6, np.abs(H), '-', linewidth=2)
    plt.plot(f*1e-6, np.abs(H2), '-', linewidth=2)
    plt.xlabel('Frequency Offset (MHz)')
    plt.ylabel('|H(f)|')
    plt.title('Channel Frequency Response')
    #plt.xlim(-BRF/2*1e-6-1, BRF/2*1e-6+1)
    #plt.xticks(np.arange(-50, 51, 25))  # ticks at -50, -25, 0, 25, 50
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'frequency_full_magnitude2.png'), dpi=300)
    plt.show()

def step5_2_TDL_full(env, rt, BRF=100e6):

    # 1) Collect the same taps as in step5_full_wideband
    c  = 299_792_458
    tx = env.emitters[0]
    rx = env.receivers[0]
    fc = rt.frequency
    lamb=c/fc

    taps = []
    # LOS
    d0   = rt.calc_distance(tx.position, rx.position)
    tau0 = d0 / c
    a0, _, _ = rt.compute_electrical_field_and_power(1.0, d0)
    taps.append((tau0, a0/d0))

    # single reflections
    for obs in env.obstacles:
        img1 = rt.compute_image_position(obs, tx.position)
        if obs.check_intersection(img1, rx.position):
            imp1 = obs.impact_point(img1, rx.position)
            if imp1 is not None:
                coeff1  = physics.calculer_coeff_reflexion(obs, tx.position, imp1)
                d1  = rt.calc_distance(img1, rx.position)
                tau1 = d1 / c
                a1, _, _ = rt.compute_electrical_field_and_power(coeff1, d1)
                taps.append((tau1, a1/d1))

    # double & triple bounces identical to step5_full_wideband…
    for obs1 in env.obstacles:
        img1 = rt.compute_image_position(obs1, tx.position)
        for obs2 in env.obstacles:
            if obs2 is obs1: continue
            img2 = rt.compute_image_position(obs2, img1)
            if obs2.check_intersection(img2, rx.position):
                imp2 = obs2.impact_point(img2, rx.position)
                if imp2 and obs1.check_intersection(img1, imp2):
                    imp1 = obs1.impact_point(img1, imp2)
                    if imp1 is not None:
                        coeff1 = physics.calculer_coeff_reflexion(obs1, tx.position, imp1)
                        coeff2 = physics.calculer_coeff_reflexion(obs2, img1, imp2)
                        coeff = coeff1*coeff2
                        d2 = rt.calc_distance(img2, rx.position)
                        tau2 = d2 / c
                        a2, _, _ = rt.compute_electrical_field_and_power(coeff, d2)
                        taps.append((tau2, a2/d2))

    for obs1 in env.obstacles:
        img1 = rt.compute_image_position(obs1, tx.position)
        for obs2 in env.obstacles:
            if obs2 is obs1: continue
            img2 = rt.compute_image_position(obs2, img1)
            for obs3 in env.obstacles:
                if obs3 in (obs1, obs2): continue
                img3 = rt.compute_image_position(obs3, img2)
                ok3 = obs3.check_intersection(img3, rx.position)
                ok2 = obs2.check_intersection(img2, obs3.impact_point(img3, rx.position) or rx.position)
                ok1 = obs1.check_intersection(img1, obs2.impact_point(img2, obs3.impact_point(img3, rx.position)) or rx.position)
                if ok1 and ok2 and ok3:
                    imp3 = obs3.impact_point(img3, rx.position)
                    imp2 = obs2.impact_point(img2, imp3)
                    imp1 = obs1.impact_point(img1, imp2)
                    if None not in (imp1, imp2, imp3):
                        coeff1 = physics.calculer_coeff_reflexion(obs1, tx.position, imp1)
                        coeff2 = physics.calculer_coeff_reflexion(obs2, img1, imp2)
                        coeff3 = physics.calculer_coeff_reflexion(obs3, img2, imp3)
                        coeff = coeff1*coeff2*coeff3
                        d3 = rt.calc_distance(img3, rx.position)
                        tau3 = d3 / c
                        a3, _, _ = rt.compute_electrical_field_and_power(coeff, d3)
                        taps.append((tau3, a3/d3))

    # 2) Build TDL taps at Δτ = 1/BRF
    dtau = 1/BRF
    max_tau = max(t for t,_ in taps)
    L = int(np.ceil(max_tau/dtau))
    l = np.arange(0, L+100)
    t_l = l * dtau

    h_l = np.zeros_like(t_l, dtype=complex)
    for tau_i, a_i in taps:
        idx = np.argmin(np.abs(t_l - tau_i))
        h_l[idx] += a_i/a0

    # 3) Plot amplitude stem of h_TDL
    plt.figure(figsize=(8,4))
    plt.plot(
        t_l*1e9,            # ns
        np.abs(lamb*h_l/(3*np.pi**2)),        # amplitude
        '-',               # line with circle markers
        linewidth=2,
        markersize=5
    )
    plt.xlabel('Delay τ (ns)')
    plt.ylabel('|$h_{TDL}(τ)$|')
    plt.title('Full Channel Tapped-Delay-Line Impulse Response')
    plt.xlim(0, t_l.max()*1e9+10)
    plt.grid(True)
    plt.tight_layout()
    fname = os.path.join(outdir, 'tdl_full_magnitude.png')
    plt.savefig(fname, dpi=300)
    plt.show()

if __name__ == '__main__':
    env = Environment()
    rt  = RayTracing(env)
    step5_full_wideband(env, rt, BRF=100e6)
    step5_2_TDL_full(env, rt, BRF=100e6)
