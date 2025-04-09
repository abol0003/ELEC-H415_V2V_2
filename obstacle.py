import numpy as np
from position import Position

class Obstacle:
    
    
    def __init__(self, start, end, material, thickness):
        self.start = start  # Doit être une instance de Position
        self.end = end      # Doit être une instance de Position
        self.material = material  # Doit être une instance de Material
        self.thickness = thickness

    def draw(self, canvas, scale=50):
        x0, y0 = self.start.get_x() * scale, self.start.get_y() * scale
        x1, y1 = self.end.get_x() * scale, self.end.get_y() * scale

        canvas.create_line(x0, y0, x1, y1, fill=self.material.color, width=self.thickness * scale)

    def is_vertical(self): #ok
        """Retourne True si l'obstacle est vertical, False sinon."""
        return self.start.x == self.end.x

    def is_horizontal(self):
        """
        Détermine si l'obstacle est horizontal.
        :return: True si horizontal, False sinon.
        """
        return self.start.y == self.end.y



    def check_intersection(self, start_point, end_point):
        """
        Vérifie si un segment donné intersecte cet obstacle.

        return: True si intersection, False sinon.
        """
        # Convertir les points de début et de fin de cet obstacle et du segment en coordonnées numpy
        p1 = np.array([self.start.x, self.start.y])
        p2 = np.array([self.end.x, self.end.y])
        q1 = np.array([start_point.x, start_point.y])
        q2 = np.array([end_point.x, end_point.y])

        # Utilisation de la formule pour vérifier l'intersection de deux segments de ligne
        r = p2 - p1
        s = q2 - q1
        rxs = np.cross(r, s)
        qpxr = np.cross((q1 - p1), r)

        if rxs == 0:
            # Les lignes sont colinéaires ou parallèles
            return False

        t = np.cross((q1 - p1), s) / rxs
        u = np.cross((q1 - p1), r) / rxs

        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection = p1 + t * r
            if np.allclose(intersection, q1) or np.allclose(intersection, q2):
                # Intersection at the ray start or end est pas compté
                return False
            return True

        return False

    def is_on_obstacle(self, intersection_point):
        """Vérifie si le point d'intersection est sur l'obstacle."""
        # Pour les segments verticaux
        if self.is_vertical():
            return min(self.start.y, self.end.y) <= intersection_point.y <= max(self.start.y, self.end.y)
        # Pour les segments horizontaux
        elif self.is_horizontal():
            return min(self.start.x, self.end.x) <= intersection_point.x <= max(self.start.x, self.end.x)
        # Pour les segments obliques
        else:
            # Utiliser l'équation paramétrique du segment
            t = (intersection_point.x - self.start.x) / (self.end.x - self.start.x)
            return 0 <= t <= 1
    def impact_point(self, ray_start, ray_end): #ok
        """
        Calcule le point d'impact entre un rayon et cet obstacle.

        return: Point d'impact, ou None si aucune intersection.
        """
        # Conversion des points en vecteurs numpy pour faciliter les opérations vectorielles
        p = np.array([self.start.x, self.start.y])
        r = np.array([self.end.x - self.start.x, self.end.y - self.start.y])
        q = np.array([ray_start.x, ray_start.y])
        s = np.array([ray_end.x - ray_start.x, ray_end.y - ray_start.y])

        # Calculs nécessaires pour trouver le point d'intersection
        r_cross_s = np.cross(r, s)

        # Si r et s sont parallèles, r_cross_s sera 0, ce qui peut indiquer des segments parallèles ou colinéaires
        # Par contre, cette fonction est appelée après une vérification d'intersection, donc cette situation ne devrait pas se présenter
        if r_cross_s == 0:
            return None


        # Calcul de t pour trouver le point d'intersection le long du segment [p, p+r]
        t = np.cross(q - p, s) / r_cross_s

        # Calcul du point d'intersection
        intersection = p + t * r
        intersection_point = Position(intersection[0], intersection[1])
        if self.is_on_obstacle(intersection_point):
            return intersection_point
        else:
            return None
    def impedance(self, frequency):
        """
        Calcule l'impédance de l'obstacle
        return: Tuple (Impédance du matériau, Impédance du vide).
        """
        mu0 = 4 * np.pi * 1e-7  # Perméabilité du vide
        eps0 = 8.854187817e-12  # Permittivité du vide
        omega = 2 * np.pi * frequency  # Fréquence angulaire
        # Impédance intrinsèque du vide
        Z0 = np.sqrt(mu0 / eps0)
        # Calcul de l'impédance du matériau de l'obstacle
        eps_m= self.material.permittivity*(10**(-9))/(36*np.pi)
        sigma = self.material.conductivity
        eps_r = eps_m - 1j * (sigma / (omega))
        # Calcul de l'impédance complexe du matériau
        Z_material = np.sqrt(mu0 / eps_r)
        return Z_material, Z0
