from position import Position

class Emitter:

    def __init__(self, position, power, frequency, gain):
        self.position = position
        self.power = power #dBm
        self.frequency = frequency
        self.gain = gain

    def draw(self, canvas, scale=50):
        """
        Dessine l'émetteur sur un canvas Tkinter.
        """
        emitter_x, emitter_y = self.position.get_x() * scale, self.position.get_y() * scale
        emitter_radius = 0.05 * scale
        canvas.create_oval(emitter_x - emitter_radius, emitter_y - emitter_radius,
                           emitter_x + emitter_radius, emitter_y + emitter_radius,
                           fill='white', outline='black')



