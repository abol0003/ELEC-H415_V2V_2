import numpy as np
from physics import transmission_totale, calculer_coeff_reflexion
from position import Position
from material import Material
from obstacle import Obstacle
class RayTracing:

    def __init__(self, environment, frequency):
        self.environment = environment
        self.frequency = frequency
        self.beta = (2 * np.pi * frequency) / 299792458
        c = 299792458
        lamb = c / frequency
        self.h_e = lamb / np.pi
        mu0 = 4 * np.pi * 1e-7
        eps0 = 8.854187817e-12
        self.Ra = 73
        Z0 = np.sqrt(mu0 / eps0)
        self.G_TX = (np.pi * Z0 * (np.abs(self.h_e)) ** 2) / (self.Ra * (lamb ** 2))
        self.P_TX = 1e-3 * (10 ** (20 / 10))



    def compute_image_position(self, obstacle, source_position):
        """
        Calcule la position image pour un obstacle donné par rapport à une position.
        """
        AB = np.array([obstacle.end.x - obstacle.start.x, obstacle.end.y - obstacle.start.y])
        n = np.array([-AB[1], AB[0]])  # Vecteur normal à l'obstacle
        A = np.array([obstacle.start.x, obstacle.start.y])
        source = np.array([source_position.x, source_position.y])
        image_position = source - 2 * (np.dot(source - A, n) / np.dot(n, n)) * n

        return Position(image_position[0], image_position[1])
    def calc_distance(self, p1, p2):
        """
        norme de la distance entre 2 points
        """
        return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

    def compute_electrical_field_and_power(self, coefficient, distance):
        """
        champs electrique (V/m) et puissance (W)
        """
        Elec_field = (coefficient * np.sqrt(60 * self.G_TX * self.P_TX) * np.exp(1j * (-distance) * self.beta)) / distance
        Power = ((self.h_e * np.abs(Elec_field)) ** 2) / (8 * self.Ra)

        return Elec_field, Power

    def direct_propagation(self, emitter, receiver):
        """
        Calcule la puissance reçue par propagation directe.
        """
        Entot, Ptot = 0, 0
        transmission_coefficient, dm = transmission_totale(self.environment.obstacles, emitter.position, receiver.position)
        distance = self.calc_distance(emitter.position, receiver.position) + dm
        if distance == 0:  # Évite la division par zéro
            distance = 10**(-6)
        E, Power = self.compute_electrical_field_and_power(transmission_coefficient, distance)

        return Power

    def reflex_and_power(self, emitter, receiver):
        """
        Calcule la puissance reçue par réflexion simple.
        """
        Power_tot = 0
        for obstacle in self.environment.obstacles:
            image_position = self.compute_image_position(obstacle, emitter.position)
            if obstacle.check_intersection(image_position, receiver.position):
                imp_p = obstacle.impact_point(image_position, receiver.position)
                transmission_coefficient1, dm_1 = transmission_totale(self.environment.obstacles, emitter.position, imp_p)
                transmission_coefficient2, dm_2 = transmission_totale(self.environment.obstacles, imp_p, receiver.position)
                reflection_coefficient = calculer_coeff_reflexion(obstacle, emitter.position, imp_p)
                coeff_tot = transmission_coefficient1 * transmission_coefficient2 * reflection_coefficient
                distance = self.calc_distance(imp_p, receiver.position) + dm_1 + dm_2
                E, Power = self.compute_electrical_field_and_power(coeff_tot, distance)
                Power_tot += Power

        return Power_tot

    def double_reflex_and_power(self, emitter, receiver):
        """
        Calcule la puissance reçue par double réflexion.
        """
        Power_tot = 0
        for obstacle1 in self.environment.obstacles:
            image_pos1 = self.compute_image_position(obstacle1, emitter.position)
            for obstacle2 in self.environment.obstacles:
                image_pos2 = self.compute_image_position(obstacle2, image_pos1)
                if obstacle1 == obstacle2:
                    continue
                elif obstacle2.check_intersection(image_pos2, receiver.position):
                    impact_point2 = obstacle2.impact_point(image_pos2, receiver.position)
                    if obstacle1.check_intersection(image_pos1, impact_point2):
                        impact_point1 = obstacle1.impact_point(image_pos1, impact_point2)
                        if impact_point1 is None or impact_point2 is None:
                            continue
                #trois coeff transmission poss car 3 rayons
                        transmission_coefficient1, dm_1 = transmission_totale(self.environment.obstacles, emitter.position, impact_point1)
                        transmission_coefficient2, dm_2 = transmission_totale(self.environment.obstacles, impact_point1, impact_point2)
                        transmission_coefficient3, dm_3 = transmission_totale(self.environment.obstacles, impact_point2, receiver.position)
                        reflection_coefficient1 = calculer_coeff_reflexion(obstacle1, emitter.position, impact_point1)
                        reflection_coefficient2 = calculer_coeff_reflexion(obstacle2, image_pos1, impact_point2)
                        coeff_tot = transmission_coefficient1 * transmission_coefficient2 * transmission_coefficient3 * reflection_coefficient1 * reflection_coefficient2
                        total_distance = self.calc_distance(emitter.position, impact_point1) + self.calc_distance(impact_point1, impact_point2) + self.calc_distance(impact_point2, receiver.position)
                        Elec, Power = self.compute_electrical_field_and_power(coeff_tot, total_distance)
                        Power_tot += Power

        return Power_tot

    def ray_tracer(self):
        """
        Exécute la simulation de ray-tracing pour tous les récepteurs et détermine la puissance maximale reçue.
        """
        for receiver in self.environment.receivers:
            max_power = -90  # Initialisation
            for emitter in self.environment.emitters:
                direct_power = self.direct_propagation(emitter, receiver)
                reflex_power = self.reflex_and_power(emitter, receiver)
                double_reflex_power = self.double_reflex_and_power(emitter, receiver)
                total_power = direct_power + reflex_power + double_reflex_power
                received_power_dBm = 10 * np.log10(total_power / 1e-3)  # Conversion en dBm
                # n'aditionne pas les valeurs pour plusieurs emetteur mais ne garde que la plus grande
                if received_power_dBm > max_power:
                    max_power = received_power_dBm
            receiver.received_power_dBm = max_power

