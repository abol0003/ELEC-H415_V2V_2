import numpy as np
from raytracing import RayTracing
from physics import calculer_angle_incidence, calculer_distance_parcourue
from environment import Environment
from position import Position


def calculate_mpc_for_all_reflections():
    # Initialisation de l'environnement et de RayTracing
    env = Environment()
    ray_tracing = RayTracing(env)

    # Parcours de tous les récepteurs et émetteurs
    for receiver in env.receivers:
        for emitter in env.emitters:
            print(f"--- Calcul des MPC pour l'émetteur {emitter.position} et le récepteur {receiver.position} ---")

            # Propagation directe (LOS)
            direct_distance = ray_tracing.calc_distance(emitter.position, receiver.position)
            direct_angle = calculer_angle_incidence(emitter.position, receiver.position, env.obstacles[0])
            direct_delay = direct_distance / 299792458 *10**(6) # Délai de propagation en secondes (vitesse de la lumière)
            print(
                f"Direct Propagation (LOS): Angle d'incidence = {np.degrees(direct_angle):.4f}°, Délai = {direct_delay:.6f}µs")

            # Réflexion simple (1 réflexion)
            for obstacle in env.obstacles:
                image_position = ray_tracing.compute_image_position(obstacle, emitter.position)
                if obstacle.check_intersection(image_position, receiver.position):
                    impact_point = obstacle.impact_point(image_position, receiver.position)
                    reflex_angle = calculer_angle_incidence(emitter.position, impact_point, obstacle)
                    reflex_distance = ray_tracing.calc_distance(image_position, receiver.position)
                    reflex_delay = reflex_distance / 299792458 *10**(6) # Délai de propagation
                    print(
                        f"Reflection 1: Angle d'incidence = {np.degrees(reflex_angle):.4f}°, Délai = {reflex_delay:.6f}µs")

            # Réflexion double (2 réflexions)
            #double_reflex_power, double_reflex_volt = ray_tracing.double_reflex_and_power(emitter, receiver)
            for obstacle1 in env.obstacles:
                for obstacle2 in env.obstacles:
                    if obstacle1 == obstacle2:
                        continue
                    image_pos1 = ray_tracing.compute_image_position(obstacle1, emitter.position)
                    image_pos2 = ray_tracing.compute_image_position(obstacle2, image_pos1)
                    if obstacle2.check_intersection(image_pos2, receiver.position):
                        impact_point2 = obstacle2.impact_point(image_pos2, receiver.position)
                    if obstacle1.check_intersection(image_pos1,impact_point2):
                        impact_point1 = obstacle1.impact_point(image_pos1,impact_point2)
                        double_reflex_angle1 = calculer_angle_incidence(emitter.position, impact_point1, obstacle1)
                        double_reflex_angle2 = calculer_angle_incidence(image_pos1, impact_point2, obstacle2)
                        double_reflex_distance = ray_tracing.calc_distance(image_pos2, receiver.position)
                        double_reflex_delay = double_reflex_distance / 299792458 *10**(6) # Délai de propagation
                        print(
                            f"Reflection 2 : angles d'incidence 1 = "
                            f"{np.degrees(double_reflex_angle1):.4f}°, "
                            f"Reflection 2 : angles d'incidence 2 = "
                            f"{np.degrees(double_reflex_angle2):.4f}°, "
                            f"délai = {double_reflex_delay:.6f} µs"
                        )

            # Réflexion triple (3 réflexions)
            #triple_reflex_power, triple_reflex_volt = ray_tracing.triple_reflex_and_power(emitter, receiver)
            for obstacle1 in env.obstacles:
                for obstacle2 in env.obstacles:
                    for obstacle3 in env.obstacles:
                        if obstacle1 == obstacle2 or obstacle2 == obstacle3:
                            continue
                        image_pos1 = ray_tracing.compute_image_position(obstacle1, emitter.position)
                        image_pos2 = ray_tracing.compute_image_position(obstacle2, image_pos1)
                        image_pos3 = ray_tracing.compute_image_position(obstacle3, image_pos2)
                        if obstacle3.check_intersection(image_pos3, receiver.position):
                            impact_point3 = obstacle3.impact_point(image_pos3, receiver.position)
                            if obstacle2.check_intersection(image_pos2, impact_point3):
                                impact_point2 = obstacle2.impact_point(image_pos2, impact_point3)
                                if obstacle1.check_intersection(image_pos1, impact_point2):
                                    impact_point1 = obstacle1.impact_point(image_pos1, impact_point2)
                                    triple_reflex_angle1 = calculer_angle_incidence(emitter.position, impact_point1,
                                                                                   obstacle1)
                                    triple_reflex_angle2 = calculer_angle_incidence(image_pos1, impact_point2,
                                                                                    obstacle2)
                                    triple_reflex_angle3 = calculer_angle_incidence(image_pos2, impact_point3,
                                                                                   obstacle3)
                                    triple_reflex_distance = ray_tracing.calc_distance(image_pos3,receiver.position)

                                    triple_reflex_delay = triple_reflex_distance / 299792458 *10**(6) # Délai de propagation
                                    print(
                                        f"Reflection 3 : angles d'incidence 1 = "
                                        f"{np.degrees(triple_reflex_angle1):.4f}°, "
                                        f"Reflection 3 : angles d'incidence 2 = "
                                        f"{np.degrees(triple_reflex_angle2):.4f}°, "
                                        f"Reflection 3 : angles d'incidence 3 = "
                                        f"{np.degrees(triple_reflex_angle3):.4f}°, "
                                        f"délai = {triple_reflex_delay:.6f} µs"
                                    )


# Exécution du calcul pour chaque MPC avec propagation directe et réflexions
calculate_mpc_for_all_reflections()
