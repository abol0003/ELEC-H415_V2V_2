import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from environment import Environment
from raytracing import RayTracing
from position import Position
import os
outdir=r'C:/Users/alexb/OneDrive - Université Libre de Bruxelles/MA1-ULBDrive/ELEC-H415/ELEC-H415_V2V/Plot'
os.makedirs(outdir, exist_ok=True)
# Speed of light (m/s)
c = 299_792_458

def compute_incidence_angle(p_src, imp_point, obstacle):
    """
    Compute the incidence angle (radians) of the ray on the obstacle.
    """
    # Vector along obstacle edge
    AB = np.array([obstacle.end.x - obstacle.start.x,
                   obstacle.end.y - obstacle.start.y])
    # Normal vector to the obstacle
    n = np.array([-AB[1], AB[0]])
    n = n / np.linalg.norm(n)
    # Vector from source to impact point
    v = np.array([imp_point.x - p_src.x,
                  imp_point.y - p_src.y])
    v = v / np.linalg.norm(v)
    # Incidence angle is angle between v and normal
    return np.arccos(np.abs(np.dot(v, n)))

def gather_mpcs(rt, emitter, receiver):
    """
    Gather delays and angles for multipath components (0 to 3 reflections).
    Returns a list of dicts with 'order', 'delay', and 'angles' (in degrees).
    """
    mpcs = []
    # Direct path (order 0)
    d0 = rt.calc_distance(emitter.position, receiver.position)
    mpcs.append({'order': 0, 'delay': d0 / c, 'angles': []})

    # Single reflection (order 1)
    for obs in rt.environment.obstacles:
        img1 = rt.compute_image_position(obs, emitter.position)
        if obs.check_intersection(img1, receiver.position):
            imp1 = obs.impact_point(img1, receiver.position)
            if imp1:
                d1 = rt.calc_distance(emitter.position, imp1)
                d2 = rt.calc_distance(imp1, receiver.position)
                angle1 = compute_incidence_angle(emitter.position, imp1, obs)
                mpcs.append({'order': 1,
                             'delay': (d1 + d2) / c,
                             'angles': np.degrees([angle1])})

    # Double reflection (order 2)
    for obs1 in rt.environment.obstacles:
        img1 = rt.compute_image_position(obs1, emitter.position)
        for obs2 in rt.environment.obstacles:
            if obs2 is obs1:
                continue
            img2 = rt.compute_image_position(obs2, img1)
            if obs2.check_intersection(img2, receiver.position):
                imp2 = obs2.impact_point(img2, receiver.position)
                if imp2 and obs1.check_intersection(img1, imp2):
                    imp1 = obs1.impact_point(img1, imp2)
                    if imp1:
                        d1 = rt.calc_distance(emitter.position, imp1)
                        d2 = rt.calc_distance(imp1, imp2)
                        d3 = rt.calc_distance(imp2, receiver.position)
                        a1 = compute_incidence_angle(emitter.position, imp1, obs1)
                        a2 = compute_incidence_angle(img1, imp2, obs2)
                        mpcs.append({'order': 2,
                                     'delay': (d1 + d2 + d3) / c,
                                     'angles': np.degrees([a1, a2])})

    # Triple reflection (order 3)
    for obs1 in rt.environment.obstacles:
        img1 = rt.compute_image_position(obs1, emitter.position)
        for obs2 in rt.environment.obstacles:
            if obs2 is obs1:
                continue
            img2 = rt.compute_image_position(obs2, img1)
            for obs3 in rt.environment.obstacles:
                if obs3 is obs2:
                    continue
                img3 = rt.compute_image_position(obs3, img2)
                if obs3.check_intersection(img3, receiver.position):
                    imp3 = obs3.impact_point(img3, receiver.position)
                    cond2 = obs2.check_intersection(img2, imp3) if imp3 else False
                    if cond2 and imp3:
                        imp2 = obs2.impact_point(img2, imp3)
                        cond1 = obs1.check_intersection(img1, imp2) if imp2 else False
                        if cond1 and imp2:
                            imp1 = obs1.impact_point(img1, imp2)
                            if imp1:
                                d1 = rt.calc_distance(emitter.position, imp1)
                                d2 = rt.calc_distance(imp1, imp2)
                                d3 = rt.calc_distance(imp2, imp3)
                                d4 = rt.calc_distance(imp3, receiver.position)
                                a1 = compute_incidence_angle(emitter.position, imp1, obs1)
                                a2 = compute_incidence_angle(img1, imp2, obs2)
                                a3 = compute_incidence_angle(img2, imp3, obs3)
                                mpcs.append({'order': 3,
                                             'delay': (d1 + d2 + d3 + d4) / c,
                                             'angles': np.degrees([a1, a2, a3])})
    return mpcs

def main():
    # Initialize environment, ray tracer, emitter, and receiver
    env = Environment()
    rt = RayTracing(env)
    emitter = env.emitters[0]
    receiver = env.receivers[0]

    mode = [0, 2]  # [MPCs, TX variation]

    # Step 1 & 2: MPCs
    if mode[0] == 1:
        mpcs = gather_mpcs(rt, emitter, receiver)
        print("MPCs (order, delay [s], angles [°]):")
        for m in mpcs:
            print(m)

    # Step 3: TX power variation
    if mode[1] == 1:
        # Define TX power range
        P_TX_vals = np.linspace(0.01, 1.0, 100)
        PRX_total2 = []
        PRX_total = []

        # Friis parameters
        G = rt.G_TX
        wavelength = c / rt.frequency
        d0 = rt.calc_distance(emitter.position, receiver.position)

        # Compute received power for each TX /!\ be carefull on the position of the emitter
        for P in P_TX_vals:
            rt.P_TX = P
            p_los, v0 = rt.direct_propagation(emitter, receiver)
            p1, v1 = rt.reflex_and_power(emitter, receiver)
            p2, v2 = rt.double_reflex_and_power(emitter, receiver)
            p3, v3 = rt.triple_reflex_and_power(emitter, receiver)
            Power = (np.abs(v0 + v1 + v2 + v3) ** 2) / (8 * rt.Ra)
            PRX_total2.append(p_los+p1+p2+p3)
            PRX_total.append(Power)  #real one taking interference

        # Compute Friis received power
        PRX_friis = P_TX_vals * G**2 * (wavelength / (4 * np.pi * d0))**2

        # Plot results
        plt.figure(figsize=(8, 5))
        plt.plot(P_TX_vals, PRX_total, label='Real Values')
        plt.plot(P_TX_vals, PRX_total2, label='Ray-Tracing ')
        plt.plot(P_TX_vals, PRX_friis, '--', label='Friis Model')
        plt.xlabel('Transmit Power $P_{TX}$ (W)')
        plt.ylabel('Received Power $P_{RX}$ (W)')
        plt.title(f'$P_{{RX}}$ vs $P_{{TX}}$ at {d0:.1f} m')
        plt.legend()
        plt.grid(alpha=0.5)
        plt.tight_layout()
        # Save figure
        plt.savefig(os.path.join(outdir, 'prx_full_vs_ptx.png')
)
        plt.show()
    # Step 3b: PRX vs distance at fixed TX power (échelle log–log)
    if mode[1] == 2:
        P_fixed = rt.P_TX
        G = rt.G_TX
        wavelength = c / rt.frequency

        # Balayage des distances (éviter 0 pour log)
        dist_vals = np.logspace(0, np.log10(1000.0), 1000)
        PRX_rt = []
        PRX_rt2=[]
        PRX_friis = []

        for d in dist_vals:
            rx_pos = Position(emitter.position.x + d, emitter.position.y)
            receiver.position = rx_pos

            p_los, v0 = rt.direct_propagation(emitter, receiver)
            p1, v1 = rt.reflex_and_power(emitter,receiver)
            p2,v2=rt.double_reflex_and_power(emitter,receiver)
            p3,v3=rt.triple_reflex_and_power(emitter,receiver)
            Power = (np.abs(v0+v1+v2+v3) ** 2) / (8 * rt.Ra)
            PRX_rt.append(10*np.log10((p_los+p1+p2+p3)/1e-3)) #averaging by raytracing
            PRX_rt2.append(10*np.log10(Power/1e-3)) #real one taking interference
            PRX_friis.append(
                10*np.log10(P_fixed * G ** 2 * (wavelength / (4 * np.pi * d)) ** 2/1e-3)
            )

        plt.figure(figsize=(8, 5))
        plt.semilogx(dist_vals, PRX_rt, label='Ray-Tracing')
        plt.semilogx(dist_vals, PRX_rt2, label='Real values')
        plt.semilogx(dist_vals, PRX_friis, '--', label='Friis Model')
        plt.xlabel('Distance (m)')
        plt.ylabel('Received Power $P_{RX}$ (dBm)')
        plt.title('$P_{RX}$ vs Distance ')
        plt.legend()
        plt.grid(which='both', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'prx_vs_distance_loglog.png'))
        plt.show()


if __name__ == '__main__':
    main()
