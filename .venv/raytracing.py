import numpy as np
from physics import transmission_totale, calculer_coeff_reflexion, frequency
from position import Position
from material import Material
from obstacle import Obstacle
from receiver import Receiver
class RayTracing:

    def __init__(self, environment):
        c = 299792458
        self.environment = environment
        self.frequency = frequency
        self.beta = (2 * np.pi * self.frequency) / c
        lamb = c / self.frequency
        self.h_e = lamb / np.pi
        mu0 = 4 * np.pi * 1e-7
        eps0 = 8.854187817e-12
        self.Ra = 73
        Z0 = np.sqrt(mu0 / eps0)
        self.G_TX = 16/(3*np.pi) #(np.pi * Z0 * (np.abs(self.h_e)) ** 2) / (self.Ra * (lamb ** 2))
        self.P_TX = 0.1 #Watt
        self.pl_exponent   = 2
        self.enableprint=False
        self.rice= False


    def compute_image_position(self, obstacle, source_position):
        """
        Calculate the image position for a given obstacle relative to a source position.
        """
        AB = np.array([obstacle.end.x - obstacle.start.x, obstacle.end.y - obstacle.start.y])
        n = np.array([-AB[1], AB[0]])  # Vecteur normal à l'obstacle
        A = np.array([obstacle.start.x, obstacle.start.y])
        source = np.array([source_position.x, source_position.y])
        image_position = source - 2 * (np.dot(source - A, n) / np.dot(n, n)) * n

        return Position(image_position[0], image_position[1])
    def calc_distance(self, p1, p2):
        """
        Compute the Euclidean distance between two points.
        """
        return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

    def compute_electrical_field_and_power(self, coefficient, distance):
        """
        Calculate the electric field (V/m), received power (W), and voltage (V) for a given
        reflection/transmission coefficient and path distance.
        """
        # if self.use_path_loss:
        #     att=(self.pl_d0 / distance) ** self.pl_exponent
        #     Elec_field = att*(coefficient * np.sqrt(60 * self.G_TX * self.P_TX) * np.exp(1j * (-distance) * self.beta)) / distance
        #     Voltage =0.5*self.h_e*Elec_field # this is VRX and not Voc
        #     Power = ((self.h_e * np.abs(Elec_field)) ** 2) / (8 * self.Ra)
        # else:
        Elec_field = coefficient * np.sqrt(60 * self.G_TX * self.P_TX) * np.exp(1j * (-distance) * self.beta) / distance
        Voltage =-0.5*self.h_e*Elec_field # this is VRX and not Voc
        Power = ((self.h_e * np.abs(Elec_field)) ** 2) / (8 * self.Ra)
        if Power > self.P_TX: Power = self.P_TX
        return Elec_field, Power, Voltage

    def direct_propagation(self, emitter, receiver):
        """
        Compute received power and voltage via direct line-of-sight propagation.
        """
        Entot, Ptot = 0, 0
        #transmission_coefficient, dm = transmission_totale(self.environment.obstacles, emitter.position, receiver.position)
        transmission_coefficient=1
        distance = self.calc_distance(emitter.position, receiver.position)
        if distance == 0:  # Avoid division by zero
            distance =0.1
        E, Power, Voltage = self.compute_electrical_field_and_power(transmission_coefficient, distance)

        return Power, Voltage

    def reflex_and_power(self, emitter, receiver):
        """
         Compute received power and voltage for single reflection paths.
         """
        Power_tot = 0
        Volt_tot = 0
        P_rice=[]

        for obstacle in self.environment.obstacles:
            image_position = self.compute_image_position(obstacle, emitter.position)
            if obstacle.check_intersection(image_position, receiver.position):
                imp_p = obstacle.impact_point(image_position, receiver.position)
                #transmission_coefficient1, dm_1 = transmission_totale(self.environment.obstacles, emitter.position, imp_p)
                #transmission_coefficient2, dm_2 = transmission_totale(self.environment.obstacles, imp_p, receiver.position)
                reflection_coefficient = calculer_coeff_reflexion(obstacle, emitter.position, imp_p)
                #coeff_tot = transmission_coefficient1 * transmission_coefficient2 * reflection_coefficient
                coeff_tot=reflection_coefficient
                distance = self.calc_distance(image_position, receiver.position)#+ dm_1 + dm_2 #or Image pos to receiver
                E, Power, Voltage = self.compute_electrical_field_and_power(coeff_tot, distance)
                if self.rice: P_rice.append(Power)
                if self.enableprint:
                    print(f"Received power after single reflection (Obstacle {obstacle}): {Power} W")
                    print(f"Received voltage after single reflection (Obstacle {obstacle}): {np.abs(Voltage)*10**6} µV and phase:{np.degrees(np.angle(Voltage))}°")
                Power_tot += Power
                Volt_tot+= Voltage
        if self.rice: return P_rice,Volt_tot
        return Power_tot, Volt_tot

    def double_reflex_and_power(self, emitter, receiver):
        """
        Compute received power and voltage for double reflection paths.
        """
        P_rice=[]
        Power_tot = 0
        Volt_tot = 0
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
                        #transmission_coefficient1, dm_1 = transmission_totale(self.environment.obstacles, emitter.position, impact_point1)
                        #transmission_coefficient2, dm_2 = transmission_totale(self.environment.obstacles, impact_point1, impact_point2)
                        #transmission_coefficient3, dm_3 = transmission_totale(self.environment.obstacles, impact_point2, receiver.position)
                        reflection_coefficient1 = calculer_coeff_reflexion(obstacle1, emitter.position, impact_point1)
                        reflection_coefficient2 = calculer_coeff_reflexion(obstacle2, image_pos1, impact_point2)
                        #coeff_tot = transmission_coefficient1 * transmission_coefficient2 * transmission_coefficient3 * reflection_coefficient1 * reflection_coefficient2
                        coeff_tot=reflection_coefficient1 * reflection_coefficient2
                        total_distance = self.calc_distance(image_pos2,receiver.position)
                        Elec, Power, Voltage = self.compute_electrical_field_and_power(coeff_tot, total_distance)
                        if self.rice: P_rice.append(Power)
                        if self.enableprint:
                            print(f"Received power after double reflection (Obstacles {obstacle1} and {obstacle2}): {Power} W")
                            print(f"Received voltage after double reflection (Obstacles {obstacle1} and {obstacle2}): {np.abs(Voltage)*10**6} µV and phase:{np.degrees(np.angle(Voltage))}°")
                        Volt_tot += Voltage
                        Power_tot += Power
        if self.rice: return P_rice,Volt_tot
        return Power_tot, Volt_tot

    def triple_reflex_and_power(self, emitter, receiver):
        """
        Compute received power and voltage for triple reflection paths.
        """
        P_rice=[]
        Power_tot = 0
        Volt_tot = 0
        for obstacle1 in self.environment.obstacles:
            image_pos1 = self.compute_image_position(obstacle1, emitter.position)
            for obstacle2 in self.environment.obstacles:
                image_pos2 = self.compute_image_position(obstacle2, image_pos1)
                if obstacle1 == obstacle2:
                    continue
                for obstacle3 in self.environment.obstacles:
                    image_pos3 = self.compute_image_position(obstacle3, image_pos2)
                    if obstacle2 == obstacle3:
                        continue

                    # Vérifie les intersections
                    if obstacle3.check_intersection(image_pos3, receiver.position):
                        impact_point3 = obstacle3.impact_point(image_pos3, receiver.position)
                        if obstacle2.check_intersection(image_pos2, impact_point3):
                            impact_point2 = obstacle2.impact_point(image_pos2, impact_point3)
                            if obstacle1.check_intersection(image_pos1, impact_point2):
                                impact_point1 = obstacle1.impact_point(image_pos1, impact_point2)
                                if impact_point1 is None or impact_point2 is None or impact_point3 is None:
                                    continue

                                # Trois coefficients de reflexion, car il y a 3 rayons
                                # transmission_coefficient1, dm_1 = transmission_totale(self.environment.obstacles,
                                #                                                       emitter.position, impact_point1)
                                # transmission_coefficient2, dm_2 = transmission_totale(self.environment.obstacles,
                                #                                                       impact_point1, impact_point2)
                                # transmission_coefficient3, dm_3 = transmission_totale(self.environment.obstacles,
                                #                                                       impact_point2, impact_point3)
                                # transmission_coefficient4, dm_4 = transmission_totale(self.environment.obstacles,
                                #                                                       impact_point3, receiver.position)

                                # Trois coefficients de réflexion, un pour chaque obstacle
                                reflection_coefficient1 = calculer_coeff_reflexion(obstacle1, emitter.position,
                                                                                   impact_point1)
                                reflection_coefficient2 = calculer_coeff_reflexion(obstacle2, image_pos1, impact_point2)
                                reflection_coefficient3 = calculer_coeff_reflexion(obstacle3, image_pos2, impact_point3)

                                # Calcul du coefficient total (transmissions et réflexions)
                                #coeff_tot = transmission_coefficient1 * transmission_coefficient2 * transmission_coefficient3 * transmission_coefficient4 * reflection_coefficient1 * reflection_coefficient2 * reflection_coefficient3
                                coeff_tot=reflection_coefficient1 * reflection_coefficient2 * reflection_coefficient3
                                # Calcul de la distance totale parcourue par le rayon
                                total_distance = self.calc_distance(image_pos3,receiver.position)
                                # Calcul du champ électrique et de la puissance
                                Elec, Power, Voltage = self.compute_electrical_field_and_power(coeff_tot, total_distance)
                                if self.rice:P_rice.append(Power)
                                if self.enableprint:
                                    print(f"Received power after triple reflection (Obstacle {obstacle1}): {Power} W")
                                    print(f"Received voltage after triple reflection (Obstacle {obstacle1}): {np.abs(Voltage)*10**6} µV and phase:{np.degrees(np.angle(Voltage))}°")
                                Volt_tot+=Voltage
                                Power_tot += Power
        if self.rice: return P_rice,Volt_tot
        return Power_tot, Volt_tot

    def ray_tracer(self):
        """
        Run the ray-tracing simulation for all receivers and record the maximum received power in dBm.
        """
        for receiver in self.environment.receivers:
            max_power = receiver.sensitivity  # Initialisation
            for emitter in self.environment.emitters:
                direct_power, direct_volt = self.direct_propagation(emitter, receiver)
                reflex_power, reflex_volt = self.reflex_and_power(emitter, receiver)
                double_reflex_power, double_reflex_volt = self.double_reflex_and_power(emitter, receiver)
                triple_reflex_power, triple_reflex_volt= self.triple_reflex_and_power(emitter, receiver)
                total_power = direct_power + reflex_power + double_reflex_power+triple_reflex_power
                total_voltage = direct_volt + reflex_volt +double_reflex_volt+triple_reflex_volt
                received_power_dBm = 10 * np.log10(total_power / 1e-3)  # Conversion en dBm
                # n'aditionne pas les valeurs pour plusieurs emetteur mais ne garde que la plus grande
                if received_power_dBm > max_power:
                    max_power = received_power_dBm
            receiver.received_power_dBm = max_power
