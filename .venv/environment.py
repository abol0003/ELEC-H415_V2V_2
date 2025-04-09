import tkinter as tk
from emitter import Emitter
from receiver import Receiver
from position import Position
from material import Material
from obstacle import Obstacle
from shapely.geometry import Polygon, Point

class Environment:
    def __init__(self):
        """
        Initialise l'environnement avec les matériaux, les obstacles, les émetteurs et les récepteurs.
        Définit également le polygone représentant la rue urbaine.
        """
        self.obstacles = []
        self.emitters = []
        self.receivers = []
        self.materials = {}
        self.init_materials()
        self.init_obstacles()
        self.init_emitters()
        self.init_receivers()
        self.street_polygon = self.create_street_polygon()

    def create_street_polygon(self):
        """
        Crée un polygone à partir des points définissant les contours de la rue urbaine.
        Utilisé pour optimiser la couverture de la rue pour les communications V2V.
        """
        points = [
            (0, 0), (20, 0), (20, 5), (15, 10), (10, 10), (10, 5), (5, 5),
            (5, 10), (0, 10)
        ]
        return Polygon(points)

    def is_inside(self, x, y):
        """
        Vérifie si un point est à l'intérieur du polygone représentant la rue.
        """
        point = Point(x, y)
        return self.street_polygon.contains(point)

    def init_materials(self):
        """
        Initialise les matériaux utilisés dans les obstacles (bâtiments, murs, etc.).
        """
        self.materials['concrete'] = Material('concrete', 6.4954, 1.43, 'gray')
        self.materials['metal'] = Material('metal', 1, 10**7, 'darkgray')
        self.materials['glass'] = Material('glass', 6.3919, 0.000107, 'lightblue')

    def init_obstacles(self):
        """
        Initialise les obstacles dans l'environnement, représentant des bâtiments, des murs, etc.
        """
        # Ajout de bâtiments en béton
        self.obstacles.append(Obstacle(Position(0, 0), Position(20, 0), self.materials['concrete'], 1.0))
        self.obstacles.append(Obstacle(Position(20, 0), Position(20, 5), self.materials['concrete'], 1.0))
        self.obstacles.append(Obstacle(Position(10, 5), Position(10, 10), self.materials['concrete'], 1.0))
        self.obstacles.append(Obstacle(Position(0, 5), Position(0, 10), self.materials['concrete'], 1.0))

        # Ajout de quelques véhicules comme obstacles métalliques
        self.obstacles.append(Obstacle(Position(4, 3), Position(6, 3), self.materials['metal'], 0.2))
        self.obstacles.append(Obstacle(Position(12, 6), Position(14, 6), self.materials['metal'], 0.2))

        # Ajout d'une baie vitrée comme obstacle en verre
        self.obstacles.append(Obstacle(Position(6, 8), Position(8, 8), self.materials['glass'], 0.05))

    def init_emitters(self):
        """
        Initialise les émetteurs dans l'environnement, qui sont les véhicules.
        """
        self.emitters.append(Emitter(Position(2, 2), 20, 60e9, 1.7))  # Véhicule émetteur 1
        self.emitters.append(Emitter(Position(15, 7), 20, 60e9, 1.7))  # Véhicule émetteur 2

    def init_receivers(self):
        """
        Initialise les récepteurs dans l'environnement.
        """
        self.receivers.append(Receiver(Position(4, 4), -90, 1.7))  # Récepteur près du premier émetteur
        self.receivers.append(Receiver(Position(12, 6), -90, 1.7))  # Récepteur près du second émetteur

    def draw(self, canvas, scale=50):
        """
        Dessine les obstacles, émetteurs et récepteurs sur un canvas Tkinter.
        """
        for obstacle in self.obstacles:
            obstacle.draw(canvas, scale)

        for emitter in self.emitters:
            emitter.draw(canvas, scale)

        for receiver in self.receivers:
            receiver.draw(canvas, scale)

def create_window_with_environment():
    """
    Crée une fenêtre Tkinter et dessine l'environnement sur un canvas.
    """
    root = tk.Tk()
    root.title("Simulation de l'Environnement Urbain V2V")
    canvas = tk.Canvas(root, width=900, height=600, background='white')
    canvas.pack(fill="both", expand=True)
    env = Environment()
    env.draw(canvas)
    root.mainloop()

#create_window_with_environment()
