import numpy as np
class Material:
    def __init__(self, name, permittivity, conductivity, color):
        self.name = name
        self.permittivity = permittivity
        self.conductivity = conductivity
        self.color = color