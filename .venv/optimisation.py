from functools import partial
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from environment import Environment
from raytracing import RayTracing
from position import Position
from receiver import Receiver
from emitter import Emitter
from heatmap import calculate_average_received_power
import time
emitter_power_cache = {}
config_cache = {}

def calc_power_at_position(position, env, ray_tracer, width, height):
    """
    Calcule la puissance à une position donnée, en utilisant un cache pour éviter les recalculs..
    """
    if position in emitter_power_cache:
        return (position, emitter_power_cache[position])
    env.emitters = [Emitter(Position(*position), power=20, frequency=60e9, gain=1.7)]
    power = calculate_average_received_power(env, ray_tracer, width, height)
    emitter_power_cache[position] = power
    return (position, power)

def eval_emitter_config(width, height, pos1, pos2, cache):
    """
    Évalue la configuration de deux émetteurs à des positions données pour calculer la puissance moyenne reçue.
    Utilise un cache pour stocker et récupérer les résultats des configurations déjà calculées pour minimiser le coût computationnel.
    """
    sorted_positions = tuple(sorted((pos1, pos2)))
    if sorted_positions in cache:
        return sorted_positions + (cache[sorted_positions],)
    env = Environment()
    env.emitters = [Emitter(Position(*pos1), power=20, frequency=60e9, gain=1.7),
                    Emitter(Position(*pos2), power=20, frequency=60e9, gain=1.7)]
    ray_tracer = RayTracing(env, 60e9)
    avg_power = calculate_average_received_power(env, ray_tracer, width, height)
    cache[sorted_positions] = avg_power
    return (pos1, pos2, avg_power)

def optimize_emitter_position_parallel(env, width, height, big_res, resolution):
    """
    Optimise la position d'un émetteur dans un environnement donné, en utilisant une grande résolution  et petite pour identifier les meilleures positions.
    Cette fonction utilise le parallélisme pour accélérer les calculs.
    """
    big_positions = [(x, y) for x in np.arange(0, width, big_res)
                        for y in np.arange(0, height, big_res) if env.is_inside(x, y)]

    ray_tracer = RayTracing(env, 60e9)
    big_func = partial(calc_power_at_position, env=env, ray_tracer=ray_tracer, width=width, height=height)

    with ProcessPoolExecutor() as executor:
        big_results = list(executor.map(big_func, big_positions))

    best_big_power = max(result[1] for result in big_results if result[1] != -np.inf)
    fine_positions = [pos for pos, power in big_results if power >= best_big_power - 0.2]

    fine_positions_expanded = [(x + dx, y + dy) for x, y in fine_positions
                               for dx in np.arange(-big_res, big_res + resolution, resolution)
                               for dy in np.arange(-big_res, big_res + resolution, resolution)
                               if env.is_inside(x + dx, y + dy)]

    fine_func = partial(calc_power_at_position, env=env, ray_tracer=ray_tracer, width=width, height=height)
    with ProcessPoolExecutor() as executor:
        fine_results = list(executor.map(fine_func, fine_positions_expanded))

    best_fine_position, best_fine_power = max(fine_results, key=lambda item: item[1])

    return best_fine_position, best_fine_power

def optimize_two_emitters_positions_parallel(base_env, width, height, big_res, fine_res):
    """
    Optimise la position de deux émetteurs en utilisant une grande résolution  puis petite.
    Stocke les résultats intermédiaires dans un dictionnaire partagé pour éviter les recalculs.
    Cette fonction utilise le parallélisme pour accélérer les calculs.
    """
    big_positions = [(x, y) for x in np.arange(0.5, width, big_res)
                        for y in np.arange(0.5, height, big_res) if base_env.is_inside(x, y)]
    manager = Manager()
    cache = manager.dict()

    big_best_power = -np.inf  # Initialisation à moins l'infini
    big_best_positions = None

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(eval_emitter_config, width, height, pos1, pos2, cache)
                   for pos1 in big_positions for pos2 in big_positions if pos1 != pos2]
        for future in futures:
            pos1, pos2, avg_power = future.result()
            if avg_power > big_best_power:
                big_best_power = avg_power
                big_best_positions = (pos1, pos2)

    fine_positions = [(x, y) for x, y in big_best_positions
                      for dx in np.arange(-big_res, big_res + fine_res, fine_res)
                      for dy in np.arange(-big_res, big_res + fine_res, fine_res)
                      if base_env.is_inside(x + dx, y + dy)]

    best_positions = None
    best_avg_power = -np.inf
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(eval_emitter_config, width, height, pos1, pos2, cache)
                   for pos1 in fine_positions for pos2 in fine_positions if pos1 != pos2]
        for future in futures:
            pos1, pos2, avg_power = future.result()
            if avg_power > best_avg_power:
                best_avg_power = avg_power
                best_positions = (pos1, pos2)

    return best_positions, best_avg_power

if __name__ == '__main__':
    env = Environment()
    start_time = time.time()
    best_position, best_power = optimize_emitter_position_parallel(env, 17, 9, 2, 0.5)
    end_time = time.time()
    print(f"Optimal Emitter Position: {best_position} with power: {best_power} dBm")
    print(f"Optimization Time: {end_time - start_time:.2f} seconds")
    start_time2 = time.time()
    best_positions, best_power = optimize_two_emitters_positions_parallel(env, 17, 9, 2 , 0.5)
    end_time2 = time.time()
    print(f"Optimal Two Emitter Positions: {best_positions} with average power: {best_power} dBm")
    print(f"Two Emitter Optimization Time: {end_time2 - start_time2:.2f} seconds")
