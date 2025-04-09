import numpy as np
from position import Position
from material import Material
from obstacle import Obstacle

frequency = 60e9
def calculer_angle_incidence(pos_emetteur, pos_incidence, obstacle):
    """
    Calcule et renvoie l'angle d'incidence avec un émetteur et un point d'incidence donnés.
    """
    if obstacle.is_vertical():
        if pos_emetteur.x == pos_incidence.x:
            theta_i = np.pi / 2
        else:
            delta_x = pos_emetteur.x - pos_incidence.x
            delta_y = pos_emetteur.y - pos_incidence.y
            theta_i = abs(np.arctan(delta_y / delta_x))
    else:  # Si l'obstacle est horizontal
        if pos_emetteur.y == pos_incidence.y:
            theta_i = np.pi / 2
        else:
            delta_x = pos_emetteur.x - pos_incidence.x
            delta_y = pos_emetteur.y - pos_incidence.y
            theta_i = abs(np.arctan(delta_x / delta_y))

    return theta_i

def calcul_angle_trans(obstacle, theta_i):
    """
    Calcule et renvoie l'angle theta_t transmis via la loi de Snell-Descartes, en considérant
    la transition entre un matériau et l'air.
    """
    # Perméabilité relative de l'air par rapport au vide
    eps_r_air = 1  # Approximation pour l'air
    # Perméabilité relative du matériau de l'obstacle
    eps_r_material = obstacle.material.permittivity
    #Snell-Decartes
    theta_t = np.arcsin(np.sin(theta_i) / np.sqrt(eps_r_material))

    return theta_t

def calculer_gamma_perp(obstacle, theta_i):
    """
    Calcule le coefficient de réflexion perpendiculaire
    """
    # Récupération des impédances Z_material et Z0 de l'objet obstacle
    Z_material, Z0 = obstacle.impedance(frequency)
    theta_t = calcul_angle_trans(obstacle, theta_i)
    gammaperp = (Z_material * np.cos(theta_i) - Z0 * np.cos(theta_t)) / (
                Z_material * np.cos(theta_i) + Z0 * np.cos(theta_t))

    return gammaperp

def calculer_gammam(obstacle):
    """
    Calcule la constante de propagation complexe.
    """
    omega=2*np.pi*frequency
    mu0 = 4 * np.pi * 1e-7
    eps_m = obstacle.material.permittivity *(10**(-9))/(36*np.pi)
    sigma_m = obstacle.material.conductivity
    alpha_m = omega * np.sqrt(mu0 * eps_m / 2) * np.sqrt(np.sqrt(1 + ((sigma_m / (omega * eps_m)) ** 2)) - 1)
    beta_m = omega * np.sqrt(mu0 * eps_m / 2) * np.sqrt(np.sqrt(1 + ((sigma_m / (omega * eps_m)) ** 2)) + 1)
    gammam= alpha_m + 1j*beta_m

    return gammam


def calculer_distance_parcourue(obstacle, theta_i):
    """
    Calcule la distance parcourue par le rayon à travers l'obstacle.
    """
    theta_t = calcul_angle_trans(obstacle, theta_i)

    return np.abs(obstacle.thickness / np.cos(theta_t))

def calculer_phase_accumulee(obstacle, theta_i):
    """
    Calcule la phase accumulée à travers l'obstacle, en utilisant la fréquence spécifiée
    pour calculer la valeur de beta du matériau de l'obstacle.
    """
    s = calculer_distance_parcourue(obstacle, theta_i)
    theta_t = calcul_angle_trans(obstacle, theta_i)
    beta_value = (2*np.pi*frequency)/299792458
    phase_accumulee = 1j*2*beta_value * s * np.sin(theta_t) * np.sin(theta_i)

    return phase_accumulee

def calculer_coeff_reflexion(obstacle, position_emetteur, position_reflexion): #ok
    """
    Calculer le coefficient de réflexion.
    """
    theta_i = calculer_angle_incidence(position_emetteur, position_reflexion, obstacle)
    theta_t =calcul_angle_trans(obstacle, theta_i)
    gammap = calculer_gamma_perp(obstacle, theta_i)
    gamma_m = calculer_gammam(obstacle)
    s = calculer_distance_parcourue(obstacle, theta_i)
    beta_s = calculer_phase_accumulee(obstacle, theta_i)
    numerateur = gammap * np.exp(-2 * gamma_m*s) * np.exp(beta_s)
    denominateur = 1 - ((gammap ** 2) * np.exp(-2 * gamma_m * s) * np.exp(beta_s))

    return gammap - ((1 - gammap ** 2)) * (numerateur / denominateur)

def calculer_coeff_transmission(obstacle, position_emetteur, position_recepteur):
    """
    Calculer le coefficient de transmission.
    """
    theta_i = calculer_angle_incidence(position_emetteur, position_recepteur, obstacle)
    s = calculer_distance_parcourue(obstacle, theta_i)
    gamma_perp = calculer_gamma_perp(obstacle, theta_i)
    gamma_m = calculer_gammam(obstacle)
    beta_s = calculer_phase_accumulee(obstacle, theta_i)

    numerateur = (1 - (gamma_perp ** 2)) * np.exp(-gamma_m * s )
    denominateur = 1 - ((gamma_perp ** 2) * np.exp(-2 * gamma_m *s) * np.exp(beta_s))

    return numerateur / denominateur


def transmission_totale(obstacles, position_depart, position_fin):
    """
    Calcule le coefficient de transmission total à travers tous les obstacles intersectés par le rayon.

    """
    coeff_total = 1.0 #si le rayon ne traverse pas d'obstacle
    distance_totale = 0
    for obstacle in obstacles:
        if obstacle.check_intersection(position_depart, position_fin):
            inter_point=obstacle.impact_point(position_depart, position_fin)
            theta_i = calculer_angle_incidence(position_depart, position_fin, obstacle)
            coeff_transmission = calculer_coeff_transmission(obstacle, position_depart, position_fin)
            coeff_total *= coeff_transmission
            distance_obstacle = calculer_distance_parcourue(obstacle, theta_i)
            distance_totale += distance_obstacle

    return coeff_total, distance_totale