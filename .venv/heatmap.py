import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import os
outdir=r'C:/Users/alexb/OneDrive - Université Libre de Bruxelles/MA1-ULBDrive/ELEC-H415/ELEC-H415_V2V/Plot'
os.makedirs(outdir, exist_ok=True)
matplotlib.use('TkAgg')
from matplotlib.collections import LineCollection
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from environment import Environment
from raytracing import RayTracing
from position import Position
from receiver import Receiver
import multiprocessing


def draw_obstacles(ax, env):
    """
    Dessine les obstacles dans l'environnement sur le graphique donné.
    Chaque obstacle est représenté par une ligne avec une couleur correspondant au matériau.
    """
    lines = []
    line_colors = []
    for obstacle in env.obstacles:
        x0, y0 = obstacle.start.x, obstacle.start.y
        x1, y1 = obstacle.end.x, obstacle.end.y
        line = [(x0, y0), (x1, y1)]
        lines.append(line)
        line_colors.append(obstacle.material.color)
    lc = LineCollection(lines, colors=line_colors, linewidths=[obstacle.thickness  for _ in lines])
    ax.add_collection(lc)
    for emitter in env.emitters:
        ax.scatter(emitter.position.x, emitter.position.y, color='white', s=10, edgecolors='black',
                   label='Emitter' if 'Emitter' not in ax.get_legend_handles_labels()[1] else "")

def dbm_to_mbps(dBm):
    """
    Convertit la puissance en dBm en débit binaire en Mbps par extrapolation linéaire.
    """
    if dBm < -90:
        return 0
    elif dBm > -40:
        return 40000
    else:
        return ((dBm + 90) * (39950 / 50) + 50)

def calculate_power_at_point(env, ray_tracer, x, y):
    dummy_receiver = Receiver(Position(x, y), sensitivity=-200)
    env.receivers = [dummy_receiver]
    ray_tracer.ray_tracer()
    return dummy_receiver.received_power_dBm #if dummy_receiver.received_power_dBm >= dummy_receiver.sensitivity else dummy_receiver.sensitivity

def calculate_average_received_power(env, ray_tracer, width, height):
    """
    Calcule la puissance moyenne reçue sur une grille représentant l'environnement.
    """
    dummy_positions = [(i, j) for i in np.arange(0, width, 0.5) for j in np.arange(0, height, 0.5) if env.is_inside(i, j)]
    total_power = 0
    count = 0
    for pos in dummy_positions:
        dummy_receiver = Receiver(Position(pos[0], pos[1]), sensitivity=-70)
        env.receivers = [dummy_receiver]
        ray_tracer.ray_tracer()
        if dummy_receiver.received_power_dBm and dummy_receiver.received_power_dBm >= -70:
            total_power += dummy_receiver.received_power_dBm
            count += 1
    return total_power / count if count else -np.inf

def create_heatmap(env, width, height, resolution, with_pl=False):
    x = np.arange(0, width, resolution)
    y = np.arange(0, height, resolution)
    X, Y = np.meshgrid(x, y)

    # Instancie le ray tracer
    rt = RayTracing(env)
    receiver=Receiver

    # Fonction pour un point
    func = partial(calculate_power_at_point, env, rt)
    with ProcessPoolExecutor() as executor:
        power = np.array(list(executor.map(func, X.ravel(), Y.ravel())))
    power_grid = power.reshape(X.shape)
    average_power = calculate_average_received_power(env, rt, width, height)
    print(f"Average Received Power: {average_power:.2f} dBm")
    title = f'Received Power Heatmap (dBm)'
    fig, ax = plt.subplots(figsize=(12, 10))
    pcm = ax.pcolormesh(X, Y, power_grid, shading='auto', cmap='viridis',
                        vmin=np.nanmin(power_grid), vmax=np.nanmax(power_grid))
    fig.colorbar(pcm, ax=ax, label='Received Power (dBm)')
    draw_obstacles(ax, env)
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    #ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"received_power_heatmap.png"), dpi=300)
    plt.show()

    # Plot débit
    # fig2, ax2 = plt.subplots(figsize=(12,10))
    # pcm2 = ax2.pcolormesh(X, Y, rate_grid, shading='auto', cmap='plasma', vmin=50, vmax=40000)
    # plt.colorbar(pcm2, ax=ax2, label='Débit Binaire (Mbps)')
    # draw_obstacles(ax2, env)
    # ax2.set(title=title2, xlabel='X (m)', ylabel='Y (m)')
    # ax2.invert_yaxis(); ax2.set_aspect('auto')
    # #plt.savefig(png2, format='jpeg')
    # plt.show()
    return power_grid

if __name__ == '__main__':
    env = Environment()
    env.emitters[0].position = Position(500, 10)
    start = time.time()
    res=1
   # multiprocessing.freeze_support()  # Windows only, safe elsewhere
    create_heatmap(env, width=1000, height=21, resolution=res,
                   with_pl=False)



    end = time.time()
    print(f"Total Heatmap Time: {end - start:.2f}s")
