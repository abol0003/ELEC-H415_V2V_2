import unittest
from environment import Environment
from raytracing import RayTracing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Change le backend à TkAgg
from itertools import cycle

class TestRayTracingWithEnvironment(unittest.TestCase):
    def setUp(self):
        self.environment = Environment()
        self.ray_tracing = RayTracing(self.environment)

    def plot_with_emitters_and_receivers(self, emitter, receiver):
        """Fonction générale pour afficher les émetteurs, récepteurs et obstacles."""
        plt.scatter(emitter.position.x, emitter.position.y, color='blue', s=100, edgecolor='black', label='Émetteur')
        plt.scatter(receiver.position.x, receiver.position.y, color='cyan', s=100, edgecolor='black', label='Récepteur')

    def plot_obstacles(self):
        """Fonction pour afficher les obstacles dans l'environnement."""
        for obstacle in self.environment.obstacles:
            plt.plot([obstacle.start.x, obstacle.end.x], [obstacle.start.y, obstacle.end.y], 'k-', linewidth=2)

    def test_direct_propagation(self):
        """Test de la propagation directe des rayons."""
        print("\nTest de la propagation directe :")
        for emitter in self.environment.emitters:
            for receiver in self.environment.receivers:
                power_received = self.ray_tracing.direct_propagation(emitter, receiver)
                print(f"Émetteur à {emitter.position}, Récepteur à {receiver.position}, Puissance reçue : {power_received} W")
                self.assertIsNotNone(power_received)
                plt.plot([emitter.position.x, receiver.position.x], [emitter.position.y, receiver.position.y], '--')

        self.plot_obstacles()
        self.plot_with_emitters_and_receivers(emitter, receiver)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Propagation directe des rayons')
        plt.show()
        plt.savefig('direct_propagation.jpeg', format='jpeg')

    def test_reflection(self):
        """Test de la réflexion simple des rayons."""
        print("\nTest de la réflexion simple :")
        colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

        for emitter in self.environment.emitters:
            for receiver in self.environment.receivers:
                power_received = self.ray_tracing.reflex_and_power(emitter, receiver)
                print(f"Émetteur à {emitter.position}, Récepteur à {receiver.position}, Puissance totale reçue après simple: {power_received} W")
                for obstacle in self.environment.obstacles:
                    color = next(colors)
                    image_position = self.ray_tracing.compute_image_position(obstacle, emitter.position)
                    if obstacle.check_intersection(image_position, receiver.position):
                        imp_p = obstacle.impact_point(image_position, receiver.position)
                        if imp_p:
                            plt.plot([emitter.position.x, imp_p.x], [emitter.position.y, imp_p.y], color+'--')
                            plt.plot([imp_p.x, receiver.position.x], [imp_p.y, receiver.position.y], color+'--')
                            power_received = self.ray_tracing.reflex_and_power(emitter, receiver)
                            print(f"Puissance reçue après réflexion (Obstacle {obstacle}): {power_received} W")

        self.plot_obstacles()
        self.plot_with_emitters_and_receivers(emitter, receiver)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Réflexion simple des rayons')
        #plt.legend()
        plt.show()
        plt.savefig('simple_reflection.jpeg', format='jpeg')

    def test_double_reflection(self):
        """Test de la réflexion double des rayons."""
        print("\nTest de la réflexion double :")
        colors = cycle(['b', 'r', 'g', 'c', 'm', 'y', 'k'])

        for emitter in self.environment.emitters:
            for receiver in self.environment.receivers:
                power_received = self.ray_tracing.double_reflex_and_power(emitter, receiver)
                print(f"Émetteur à {emitter.position}, Récepteur à {receiver.position}, Puissance totale reçue après réflexion double : {power_received} W")
                self.assertIsNotNone(power_received)
                #self.assertGreater(power_received, 0)

                for obstacle1 in self.environment.obstacles:
                    image_pos1 = self.ray_tracing.compute_image_position(obstacle1, emitter.position)
                    for obstacle2 in self.environment.obstacles:
                        if obstacle1 == obstacle2:
                            continue
                        image_pos2 = self.ray_tracing.compute_image_position(obstacle2, image_pos1)
                        if obstacle2.check_intersection(image_pos2, receiver.position):
                            impact_point2 = obstacle2.impact_point(image_pos2, receiver.position)
                            if obstacle1.check_intersection(image_pos1, impact_point2):
                                impact_point1 = obstacle1.impact_point(image_pos1, impact_point2)
                                if impact_point1 is not None and impact_point2 is not None:
                                    color = next(colors)
                                    plt.plot([emitter.position.x, impact_point1.x], [emitter.position.y, impact_point1.y], color + '--')
                                    plt.plot([impact_point1.x, impact_point2.x], [impact_point1.y, impact_point2.y], color + '--')
                                    plt.plot([impact_point2.x, receiver.position.x], [impact_point2.y, receiver.position.y], color + '--')

        self.plot_obstacles()
        self.plot_with_emitters_and_receivers(emitter, receiver)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Doubles réflexions des rayons')
        plt.show()
        plt.savefig('double_reflection.jpeg', format='jpeg')

    def test_triple_reflection(self):
        """Test de la réflexion triple des rayons."""
        print("\nTest de la réflexion triple :")
        colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

        for emitter in self.environment.emitters:
            for receiver in self.environment.receivers:
                power_received = self.ray_tracing.triple_reflex_and_power(emitter, receiver)
                print(f"Émetteur à {emitter.position}, Récepteur à {receiver.position}, Puissance totale reçue après réflexion triple : {power_received} W")
                self.assertIsNotNone(power_received)
                #self.assertGreater(power_received, 0)

                for obstacle1 in self.environment.obstacles:
                    image_pos1 = self.ray_tracing.compute_image_position(obstacle1, emitter.position)
                    for obstacle2 in self.environment.obstacles:
                        if obstacle1 == obstacle2:
                            continue
                        image_pos2 = self.ray_tracing.compute_image_position(obstacle2, image_pos1)
                        for obstacle3 in self.environment.obstacles:
                            if obstacle3 == obstacle2:
                                continue
                            image_pos3 = self.ray_tracing.compute_image_position(obstacle3, image_pos2)
                            if obstacle3.check_intersection(image_pos3, receiver.position):
                                impact_point3 = obstacle3.impact_point(image_pos3, receiver.position)
                                if obstacle2.check_intersection(image_pos2, impact_point3):
                                    impact_point2 = obstacle2.impact_point(image_pos2, impact_point3)
                                    if obstacle1.check_intersection(image_pos1, impact_point2):
                                        impact_point1 = obstacle1.impact_point(image_pos1, impact_point2)
                                        if impact_point1 is not None and impact_point2 is not None and impact_point3 is not None:
                                            color = next(colors)
                                            plt.plot([emitter.position.x, impact_point1.x], [emitter.position.y, impact_point1.y], color + '--')
                                            plt.plot([impact_point1.x, impact_point2.x], [impact_point1.y, impact_point2.y], color + '--')
                                            plt.plot([impact_point2.x, impact_point3.x], [impact_point2.y, impact_point3.y], color + '--')
                                            plt.plot([impact_point3.x, receiver.position.x], [impact_point3.y, receiver.position.y], color + '--')

        self.plot_obstacles()
        self.plot_with_emitters_and_receivers(emitter, receiver)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Triple réflexion des rayons')
        plt.show()
        plt.savefig('triple_reflection.jpeg', format='jpeg')

if __name__ == '__main__':
    unittest.main()
