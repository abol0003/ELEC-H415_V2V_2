from position import Position

class Receiver:
    def __init__(self, position, sensitivity, gain):
        self.position = position
        self.sensitivity = sensitivity
        self.gain = gain
        self.received_power = None

    def draw(self, canvas, scale=50):
        """
        Dessine le récepteur sur un canvas Tkinter.
        """
        x, y = self.position.get_x(), self.position.get_y()
        receiver_radius = 0.05 * scale  # Taille visuelle du récepteur
        canvas.create_oval(x * scale - receiver_radius, y * scale - receiver_radius,
                           x * scale + receiver_radius, y * scale + receiver_radius,
                           fill='green', outline='black')
        if self.received_power is not None:
            # Affiche la puissance reçue à côté du récepteur
            canvas.create_text(x * scale + receiver_radius + 10, y * scale,
                               text=f"{self.received_power:.2f} dBm", fill="blue")
