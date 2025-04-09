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
        Définit également le polygone représentant l'appartement.
        """
        self.obstacles = []
        self.emitters = []
        self.receivers = []
        self.materials = {}
        self.init_materials()
        self.init_obstacles()
        self.init_emitters()
        self.init_receivers()
        self.appart_polygon = self.create_appart_polygon()

    def create_appart_polygon(self):
        """
        Crée un polygone à partir des points définissant les contours de l'appartement.
        Utilisé pour optimisation de la couverture d'appartement.'
        """
        points = [
            (0, 0), (15, 0), (15, 4), (12, 8), (9, 8), (9, 6), (6, 6),
            (6, 8), (0, 8)
        ]
        return Polygon(points)

    def is_inside(self, x, y):
        """
        Vérifie si un point est à l'intérieur du polygone de l'appartement.
        """
        point = Point(x, y)
        return self.appart_polygon.contains(point)

    def init_materials(self):
        """
        Initialise les matériaux utilisés dans les obstacles.
        """
        self.materials['concrete'] = Material('concrete', 6.4954, 1.43, 'black')
        self.materials['cloison'] = Material('cloison', 2.7, 0.005346, 'red')
        self.materials['glass'] = Material('glass', 6.3919, 0.000107, 'lightblue')
        self.materials['metal'] = Material('metal', 1, 10**7, 'grey')

    def init_obstacles(self):
        """
        Initialise les obstacles dans l'environnement, utilisant les matériaux définis.
        """
        # Murs en béton
        self.obstacles.append(Obstacle(Position(0, 0), Position(15, 0), self.materials['concrete'], 0.3))
        self.obstacles.append(Obstacle(Position(15, 0), Position(15, 4), self.materials['concrete'], 0.3))
        self.obstacles.append(Obstacle(Position(7, 0), Position(7, 4), self.materials['concrete'], 0.3))
        self.obstacles.append(Obstacle(Position(0, 0), Position(0, 8), self.materials['concrete'], 0.3))
        self.obstacles.append(Obstacle(Position(0, 8), Position(6, 8), self.materials['concrete'], 0.3))
        self.obstacles.append(Obstacle(Position(4, 8), Position(4, 6), self.materials['concrete'], 0.3))
        self.obstacles.append(Obstacle(Position(4, 6), Position(9, 6), self.materials['concrete'], 0.3))
        self.obstacles.append(Obstacle(Position(9, 6), Position(9, 8), self.materials['concrete'], 0.3))
        self.obstacles.append(Obstacle(Position(9, 8), Position(12, 8), self.materials['concrete'], 0.3))

        # Cloisons
        self.obstacles.append(Obstacle(Position(4, 0), Position(4, 4), self.materials['cloison'], 0.1))
        self.obstacles.append(Obstacle(Position(4, 4), Position(5, 4), self.materials['cloison'], 0.1))
        self.obstacles.append(Obstacle(Position(0, 5), Position(4, 5), self.materials['cloison'], 0.1))
        self.obstacles.append(Obstacle(Position(6, 4), Position(9, 4), self.materials['cloison'], 0.1))
        self.obstacles.append(Obstacle(Position(11, 0), Position(11, 4), self.materials['cloison'], 0.1))
        self.obstacles.append(Obstacle(Position(11, 4), Position(10, 4), self.materials['cloison'], 0.1))

        # Baie vitrée
        self.obstacles.append(Obstacle(Position(12, 8), Position(15, 4), self.materials['glass'], 0.05))

        # Ascenseur en métal
        self.obstacles.append(Obstacle(Position(4.25, 6.25), Position(5.755, 6.25), self.materials['metal'], 0.05))
        self.obstacles.append(Obstacle(Position(5.755, 6.25), Position(5.755, 7.755), self.materials['metal'], 0.05))
        self.obstacles.append(Obstacle( Position(5.775, 7.755), Position(4.25, 7.755), self.materials['metal'], 0.05))
        self.obstacles.append(Obstacle(Position(4.25, 7.775), Position(4.25, 6.25), self.materials['metal'], 0.05))
        #porte metalique
        self.obstacles.append(Obstacle(Position(5.8, 7.875), Position(5.8, 6.15), self.materials['metal'], 0.05))

    def init_emitters(self):
        """
        Initialise les émetteurs dans l'environnement.
        """
        self.emitters.append(Emitter(Position(9.4, 7), 20, 60e9, 1.7))
        self.emitters.append(Emitter(Position(7, 4.5), 20, 60e9, 1.7)) # emitteur optimal avec et sans ascenseur
        self.emitters.append(Emitter(Position(2.5, 2.5), 20, 60e9, 1.7)) # emitteur optimale pour 2 emetteur avec et sans ascenseur
        self.emitters.append(Emitter(Position(10.5, 4.5), 20, 60e9, 1.7)) # emitteur optimale pour 2 emetteur avec ascenseur
        self.emitters.append(Emitter(Position(8.5, 4.5), 20, 60e9, 1.7)) # emitteur optimale pour 2 emetteur sans ascenseur

    def init_receivers(self):
        """
        Initialise les émetteurs dans l'environnement.
        Utile pour valider à l'aide de l'ex4.1
        """
        #self.receivers.append(Receiver(Position(47, 65), -90, 1.7))

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
    root.title("Simulation de l'Environnement")
    canvas = tk.Canvas(root, width=900, height=600, background='white')
    canvas.pack(fill="both", expand=True)
    env = Environment()
    env.draw(canvas)
    root.mainloop()
#create_window_with_environment()