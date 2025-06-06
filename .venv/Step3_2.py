import unittest
from environment import Environment
from raytracing import RayTracing
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
outdir=r'C:/Users/alexb/OneDrive - Université Libre de Bruxelles/MA1-ULBDrive/ELEC-H415/ELEC-H415_V2V/Plot'
os.makedirs(outdir, exist_ok=True)
matplotlib.use('TkAgg')  # Use TkAgg backend for display
from itertools import cycle

class TestRayTracingWithEnvironment(unittest.TestCase):
    def setUp(self):
        self.environment = Environment()
        self.ray_tracing = RayTracing(self.environment)
        self.ray_tracing.enableprint=True #to avoid lot of print
        self.colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

    def plot_with_emitters_and_receivers(self, emitter, receiver):
        plt.scatter(emitter.position.x, emitter.position.y,
                    color='blue', s=100, edgecolor='black', label='Emitter')
        plt.scatter(receiver.position.x, receiver.position.y,
                    color='cyan', s=100, edgecolor='black', label='Receiver')

    def plot_obstacles(self):
        for obstacle in self.environment.obstacles:
            plt.plot([obstacle.start.x, obstacle.end.x],
                     [obstacle.start.y, obstacle.end.y], 'k-', linewidth=2)

    def _plot_segments(self, segments):
        for (p1, p2), color in segments:
            plt.plot([p1.x, p2.x], [p1.y, p2.y], color + '--')

    def _finalize_plot(self, title, filename=None):
        self.plot_obstacles()
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(title)
        plt.legend(loc='best', fontsize='small')
        plt.show()
        if filename:
            plt.savefig(filename, format='jpeg')

    def test_direct_propagation(self):
        print("\nTesting direct line-of-sight propagation:")
        for emitter in self.environment.emitters:
            for receiver in self.environment.receivers:
                power, volt = self.ray_tracing.direct_propagation(emitter, receiver)
                V=np.abs(volt)
                phi=np.degrees(np.angle(volt))
                print(f"Emitter at {emitter.position}, Receiver at {receiver.position}, "
                      f"Received LOS power: {power} W")
                print(f"Received voltage: {V*(10**6)} µV with phase:{phi}°")
                self.assertIsNotNone(power)
                segments = [((emitter.position, receiver.position), next(self.colors))]
                self.plot_with_emitters_and_receivers(emitter, receiver)
                self._plot_segments(segments)
        self._finalize_plot('Direct Ray Propagation')
        #plt.savefig(os.path.join(outdir, f"direct_prop.png"))

    def test_reflection(self):
        print("\nTesting single reflection propagation:")
        for emitter in self.environment.emitters:
            for receiver in self.environment.receivers:
                p, v = self.ray_tracing.reflex_and_power(emitter, receiver)
                V = np.abs(v)
                phi = np.degrees(np.angle(v))
                print(f"Emitter at {emitter.position}, Receiver at {receiver.position}, "
                      f"Power after single reflection: {p} W")
                print(f"Received voltage: {V*(10**6)} µV with phase:{phi}°")
                self.assertIsNotNone(p)
                segments = []
                for obstacle in self.environment.obstacles:
                    img = self.ray_tracing.compute_image_position(obstacle, emitter.position)
                    if obstacle.check_intersection(img, receiver.position):
                        imp = obstacle.impact_point(img, receiver.position)
                        if imp:
                            c = next(self.colors)
                            segments.append(((emitter.position, imp), c))
                            segments.append(((imp, receiver.position), c))
                self.plot_with_emitters_and_receivers(emitter, receiver)
                self._plot_segments(segments)
        self._finalize_plot('Single Reflection Rays')

    def test_double_reflection(self):
        print("\nTesting double reflection propagation:")
        for emitter in self.environment.emitters:
            for receiver in self.environment.receivers:
                p, v = self.ray_tracing.double_reflex_and_power(emitter, receiver)
                V = np.abs(v)
                phi = np.degrees(np.angle(v))
                print(f"Emitter at {emitter.position}, Receiver at {receiver.position}, "
                      f"Power after double reflection: {p} W")
                print(f"Received voltage: {V*(10**6)} µV with phase:{phi}°")
                self.assertIsNotNone(p)
                segments = []
                for o1 in self.environment.obstacles:
                    img1 = self.ray_tracing.compute_image_position(o1, emitter.position)
                    for o2 in self.environment.obstacles:
                        if o1 == o2:
                            continue
                        img2 = self.ray_tracing.compute_image_position(o2, img1)
                        if o2.check_intersection(img2, receiver.position):
                            imp2 = o2.impact_point(img2, receiver.position)
                            if o1.check_intersection(img1, imp2):
                                imp1 = o1.impact_point(img1, imp2)
                                if imp1 and imp2:
                                    c = next(self.colors)
                                    segments.extend([
                                        ((emitter.position, imp1), c),
                                        ((imp1, imp2), c),
                                        ((imp2, receiver.position), c)
                                    ])
                self.plot_with_emitters_and_receivers(emitter, receiver)
                self._plot_segments(segments)
        self._finalize_plot('Double Reflection Rays')

    def test_triple_reflection(self):
        print("\nTesting triple reflection propagation:")
        for emitter in self.environment.emitters:
            for receiver in self.environment.receivers:
                p, v = self.ray_tracing.triple_reflex_and_power(emitter, receiver)
                V = np.abs(v)
                phi = np.degrees(np.angle(v))
                print(f"Emitter at {emitter.position}, Receiver at {receiver.position}, "
                      f"Power after triple reflection: {p} W")
                print(f"Received voltage: {V*(10**6)} µV with phase:{phi}°")
                self.assertIsNotNone(p)
                segments = []
                for o1 in self.environment.obstacles:
                    img1 = self.ray_tracing.compute_image_position(o1, emitter.position)
                    for o2 in self.environment.obstacles:
                        if o1 == o2:
                            continue
                        img2 = self.ray_tracing.compute_image_position(o2, img1)
                        for o3 in self.environment.obstacles:
                            if o3 == o2:
                                continue
                            img3 = self.ray_tracing.compute_image_position(o3, img2)
                            if o3.check_intersection(img3, receiver.position):
                                imp3 = o3.impact_point(img3, receiver.position)
                                if o2.check_intersection(img2, imp3):
                                    imp2 = o2.impact_point(img2, imp3)
                                    if o1.check_intersection(img1, imp2):
                                        imp1 = o1.impact_point(img1, imp2)
                                        if imp1 and imp2 and imp3:
                                            c = next(self.colors)
                                            segments.extend([
                                                ((emitter.position, imp1), c),
                                                ((imp1, imp2), c),
                                                ((imp2, imp3), c),
                                                ((imp3, receiver.position), c)
                                            ])
                self.plot_with_emitters_and_receivers(emitter, receiver)
                self._plot_segments(segments)
        self._finalize_plot('Triple Reflection Rays')

    def test_combined_reflections(self):
        print("\nTesting combined LOS and reflections:")
        for emitter in self.environment.emitters:
            for receiver in self.environment.receivers:
                p_los, v_los = self.ray_tracing.direct_propagation(emitter, receiver)
                p1, v1 = self.ray_tracing.reflex_and_power(emitter, receiver)
                p2, v2 = self.ray_tracing.double_reflex_and_power(emitter, receiver)
                p3, v3 = self.ray_tracing.triple_reflex_and_power(emitter, receiver)
                p_total = p_los + p1 + p2 + p3
                v_total = v_los + v1 + v2 + v3
                V=np.abs(v_total)
                phi=np.degrees(np.angle(v_total))
                print(f"Emitter {emitter.position}, Receiver {receiver.position} -> "
                      f"Total power = {p_total:.6e} W, Total voltage = {V*(10**6):.4f} µV with phase:{phi:.4f} °.")
                self.assertIsNotNone(p_total)
                self.assertIsNotNone(v_total)
                self.assertGreaterEqual(p_total, p_los)

if __name__ == '__main__':
    suite = unittest.TestSuite([
        TestRayTracingWithEnvironment('test_direct_propagation'),
        TestRayTracingWithEnvironment('test_reflection'),
        TestRayTracingWithEnvironment('test_double_reflection'),
        TestRayTracingWithEnvironment('test_triple_reflection'),
        TestRayTracingWithEnvironment('test_combined_reflections'),
    ])
    runner = unittest.TextTestRunner()
    runner.run(suite)