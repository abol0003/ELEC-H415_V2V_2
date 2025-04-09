import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.collections import LineCollection
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from environment import Environment
from raytracing import RayTracing
from position import Position
from receiver import Receiver
import time

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
    """
    Calcule la puissance de signal reçue en dBm à un point spécifique.
    Utilise un récepteur fictif pour mesurer la puissance.
    """
    dummy_receiver = Receiver(Position(x, y), sensitivity=-70)
    env.receivers = [dummy_receiver]
    ray_tracer.ray_tracer()
    return dummy_receiver.received_power_dBm if dummy_receiver.received_power_dBm >= -90 else -90

def calculate_average_received_power(env, ray_tracer, width, height):
    """
    Calcule la puissance moyenne reçue sur une grille représentant l'environnement.
    """
    dummy_positions = [(i, j) for i in np.arange(0, width, 0.5) for j in np.arange(0, height, 0.5) if env.is_inside(i, j)]
    total_power = 0
    count = 0
    for pos in dummy_positions:
        dummy_receiver = Receiver(Position(pos[0], pos[1]), sensitivity=-90, gain=1.7)
        env.receivers = [dummy_receiver]
        ray_tracer.ray_tracer()
        if dummy_receiver.received_power_dBm and dummy_receiver.received_power_dBm >= -90:
            total_power += dummy_receiver.received_power_dBm
            count += 1
    return total_power / count if count else -np.inf

def create_heatmap(env, width, height, resolution):
    """
    Crée une heatmap de la puissance reçue et du débit binaire sur une grille définie par width et height avec la résolution donnée.
    Utilise le multiprocessing pour calculer la puissance en chaque point de la grille.
    """
    x = np.arange(0, width, resolution)
    y = np.arange(0, height, resolution)
    X, Y = np.meshgrid(x, y)
    power_grid = np.full(X.shape, np.nan)
    rate_grid = np.full(X.shape, np.nan)
    ray_tracer = RayTracing(env)
    func = partial(calculate_power_at_point, env, ray_tracer)
    with ProcessPoolExecutor() as executor:
        results = executor.map(func, X.ravel(), Y.ravel())
    power_grid = np.array(list(results)).reshape(X.shape)
    rate_grid = np.vectorize(dbm_to_mbps)(power_grid)
    #average_power = calculate_average_received_power(env, ray_tracer, width, height)
    #print(f"Average Received Power: {average_power:.2f} dBm")
    #plot des heatmaps
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    power_heatmap = ax1.pcolormesh(X, Y, power_grid, shading='auto', cmap='viridis', vmin=-90, vmax=-40)
    plt.colorbar(power_heatmap, ax=ax1, label='Puissance Reçue (dBm)')
    draw_obstacles(ax1, env)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Heatmap de la Puissance Reçue en dBm')
    ax1.set_xlim([0, width])
    ax1.set_ylim([0, height])
    ax1.set_aspect('auto')
    ax1.invert_yaxis()
    ax1.set_xticks(np.arange(0, width, resolution))  # Résolution 5x5 ou comme définie
    ax1.set_yticks(np.arange(0, height, resolution))  # Résolution 5x5 ou comme définie
    ax1.grid(True)  # Grille activée avec les ticks calculés automatiquement
    plt.savefig('dBmheat.jpeg', format='jpeg')
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(12, 10))
    rate_heatmap = ax2.pcolormesh(X, Y, rate_grid, shading='auto', cmap='plasma', vmin=50, vmax=40000)
    plt.colorbar(rate_heatmap, ax=ax2, label='Débit Binaire (Mbps)')
    draw_obstacles(ax2, env)
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Heatmap du Débit Binaire en Mbps')
    ax2.set_xlim([0, width])
    ax2.set_ylim([0, height])
    ax2.set_aspect('auto')
    ax2.invert_yaxis()
    ax2.set_xticks(np.arange(0, width, resolution))  
    ax2.set_yticks(np.arange(0, height, resolution)) 
    ax2.grid(True)
    plt.savefig('Mbpsheat.jpeg', format='jpeg')
    plt.show()

if __name__ == '__main__':
    env = Environment()
    start_time = time.time()
    create_heatmap(env, width=1000, height=25, resolution=5)
    end_time = time.time()
    print(f"Heatmap Time: {end_time - start_time:.2f} seconds")
