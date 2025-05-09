
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from environment import Environment
from raytracing import RayTracing

# Vitesse de la lumière
c = 299_792_458


def compute_incidence_angle(p_src, imp_point, obstacle):
    """Calcule l'angle d'incidence (rad) du rayon sur l'obstacle"""
    AB = np.array([obstacle.end.x - obstacle.start.x,
                   obstacle.end.y - obstacle.start.y])
    n = np.array([-AB[1], AB[0]])
    n = n / np.linalg.norm(n)
    v = np.array([imp_point.x - p_src.x,
                  imp_point.y - p_src.y])
    v = v / np.linalg.norm(v)
    return np.arccos(np.abs(np.dot(v, n)))


def gather_mpcs(rt, emitter, receiver):
    """
    Rassemble délais et angles pour les MPC (0 à 3 réflexions).
    Retourne liste de dicts: 'order','delay','angles'.
    """
    mpcs = []
    # Direct
    d0 = rt.calc_distance(emitter.position, receiver.position)
    mpcs.append({'order': 0, 'delay': d0 / c, 'angles': []})

    # 1 réflexion
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

    # 2 réflexions
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

    # 3 réflexions
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
                    cond1 = False
                    if cond2 and imp3:
                        imp2 = obs2.impact_point(img2, imp3)
                        cond1 = obs1.check_intersection(img1, imp2) if imp2 else False
                    if cond1 and imp3 and imp2:
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
    # Initialisation
    env = Environment()
    rt = RayTracing(env)
    emitter = env.emitters[0]
    receiver = env.receivers[0]
    mode=[0, 1]
    # Étape 1 & 2 : MPCs
    if mode[0]==1:
        mpcs = gather_mpcs(rt, emitter, receiver)
        print("MPCs (ordre, délai [s], angles [°]):")
        for m in mpcs:
            print(m)

    # Étape 3 : variation de PTX
    if mode[1]==1:
        P_TX_vals = np.linspace(0.01, 1.0, 100)
        PRX_total = []
        # Paramètres Friis
        G = rt.G_TX
        lamb = c / rt.frequency
        d0 = rt.calc_distance(emitter.position, receiver.position)

        for P in P_TX_vals:
            rt.P_TX = P
            p0, _ = rt.direct_propagation(emitter, receiver)
           # p1, _ = rt.reflex_and_power(emitter, receiver)
           # p2, _ = rt.double_reflex_and_power(emitter, receiver)
           # p3, _ = rt.triple_reflex_and_power(emitter, receiver)
            #PRX_total.append(p0 + p1 + p2 + p3)
            PRX_total.append(p0)


        PRX_friis = P_TX_vals * G**2 * (lamb / (4 * np.pi * d0))**2

        # Tracé
        plt.figure(figsize=(8, 5))
        plt.plot(P_TX_vals, PRX_total, label='Ray-tracing (0–3 réflexions)')
        plt.plot(P_TX_vals, PRX_friis, '--', label='Friis')
        plt.xlabel('$P_{TX}$ (W)')
        plt.ylabel('$P_{RX}$ (W)')
        plt.title(f"Step 3.3 – $P_{{RX}}(P_{{TX}})$ full channel vs Friis à {d0:.1f} m")
        plt.legend()
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()
        outfile = 'prx_full_vs_ptx.png'
        plt.savefig(outfile)
        print(f"Figure enregistrée : {outfile}")

if __name__ == '__main__':
    main()
