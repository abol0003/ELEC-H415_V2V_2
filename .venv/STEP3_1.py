import numpy as np
from raytracing import RayTracing
from physics import calculer_angle_incidence, calculer_distance_parcourue
from environment import Environment
from position import Position

def calculate_mpc_for_all_reflections():
    # Initialize environment and ray tracer
    env = Environment()
    ray_tracing = RayTracing(env)

    # Loop through all receivers and emitters
    for receiver in env.receivers:
        for emitter in env.emitters:
            print(f"--- Computing MPCs for emitter at {emitter.position} and receiver at {receiver.position} ---")

            # Direct path (LOS)
            direct_distance = ray_tracing.calc_distance(emitter.position, receiver.position)
            direct_angle = calculer_angle_incidence(emitter.position, receiver.position, env.obstacles[0])
            # Propagation delay in microseconds
            direct_delay = direct_distance / 299_792_458 * 1e6
            print(
                f"Direct Path: Incidence angle = {np.degrees(direct_angle):.4f}°, Delay = {direct_delay:.6f} µs"
            )

            # Single reflection (order 1)
            for obstacle in env.obstacles:
                image_pos = ray_tracing.compute_image_position(obstacle, emitter.position)
                if obstacle.check_intersection(image_pos, receiver.position):
                    impact_point = obstacle.impact_point(image_pos, receiver.position)
                    reflex_angle = calculer_angle_incidence(emitter.position, impact_point, obstacle)
                    reflex_distance = ray_tracing.calc_distance(image_pos, receiver.position)
                    reflex_delay = reflex_distance / 299_792_458 * 1e6
                    print(
                        f"Reflection 1: Incidence angle = {np.degrees(reflex_angle):.4f}°, Delay = {reflex_delay:.6f} µs"
                    )

            # Double reflection (order 2)
            for obs1 in env.obstacles:
                for obs2 in env.obstacles:
                    if obs1 is obs2:
                        continue
                    img1 = ray_tracing.compute_image_position(obs1, emitter.position)
                    img2 = ray_tracing.compute_image_position(obs2, img1)
                    if obs2.check_intersection(img2, receiver.position):
                        imp2 = obs2.impact_point(img2, receiver.position)
                        if obs1.check_intersection(img1, imp2):
                            imp1 = obs1.impact_point(img1, imp2)
                            angle1 = calculer_angle_incidence(emitter.position, imp1, obs1)
                            angle2 = calculer_angle_incidence(img1, imp2, obs2)
                            distance2 = ray_tracing.calc_distance(img2, receiver.position)
                            delay2 = distance2 / 299_792_458 * 1e6
                            print(
                                f"Reflection 2: Incidence angles = {np.degrees(angle1):.4f}°, "
                                f"{np.degrees(angle2):.4f}°, Delay = {delay2:.6f} µs"
                            )

            # Triple reflection (order 3)
            for obs1 in env.obstacles:
                for obs2 in env.obstacles:
                    for obs3 in env.obstacles:
                        if obs1 is obs2 or obs2 is obs3:
                            continue
                        img1 = ray_tracing.compute_image_position(obs1, emitter.position)
                        img2 = ray_tracing.compute_image_position(obs2, img1)
                        img3 = ray_tracing.compute_image_position(obs3, img2)
                        if obs3.check_intersection(img3, receiver.position):
                            imp3 = obs3.impact_point(img3, receiver.position)
                            if obs2.check_intersection(img2, imp3):
                                imp2 = obs2.impact_point(img2, imp3)
                                if obs1.check_intersection(img1, imp2):
                                    imp1 = obs1.impact_point(img1, imp2)
                                    angle1 = calculer_angle_incidence(emitter.position, imp1, obs1)
                                    angle2 = calculer_angle_incidence(img1, imp2, obs2)
                                    angle3 = calculer_angle_incidence(img2, imp3, obs3)
                                    distance3 = ray_tracing.calc_distance(img3, receiver.position)
                                    delay3 = distance3 / 299_792_458 * 1e6
                                    print(
                                        f"Reflection 3: Incidence angles = {np.degrees(angle1):.4f}°, "
                                        f"{np.degrees(angle2):.4f}°, {np.degrees(angle3):.4f}°, "
                                        f"Delay = {delay3:.6f} µs"
                                    )

# Execute MPC computation for all reflection orders
if __name__ == '__main__':
    calculate_mpc_for_all_reflections()
