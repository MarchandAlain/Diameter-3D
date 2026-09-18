"""
maxdist_minidisk_model.py — calcule le diametre exact (paire de points les
plus eloignes, distance au carre) d'un nuage de points en 2D ou 3D euclidien.

Pipeline : MiniDisk calcule d'abord le plus petit cercle/sphere englobant
(MEC). Si son support est une paire de points, cette paire EST le diametre
(reponse exacte, MaxDist inutile). Sinon, MaxDist recherche le diametre par
amelioration successive, aide par trois mecanismes bases sur le MEC :
  - filtre radial : elimine definitivement tout point P a distance <= 2r-R
    du centre MEC (r = demi-diametre courant, R = rayon MEC) - un tel point
    ne peut plus ameliorer le diametre (inegalite triangulaire).
  - invariant local : quand un point C est completement elimine, tout point
    F a distance <= eps de C l'est aussi (meme argument), eps derive de la
    plus grande distance dejà vue entre C et un point vivant.
  - rappel radial periodique : reapplique le filtre radial en cours de
    recherche (le diametre courant s'ameliore, donc le rayon d'elimination
    aussi), avec un espacement en backoff exponentiel pour eviter les
    rappels steriles.
"""

import math
from itertools import combinations
from typing import Optional


# =============================================================== geometrie de base

def distance(A: tuple, B: tuple, dimension: int) -> float:
    """Carre de la distance euclidienne entre A et B."""
    return sum((A[i] - B[i]) ** 2 for i in range(dimension))


def ball(A: tuple, B: tuple, dimension: int):
    """Boule de diametre AB. Retourne (centre, rayon^2)."""
    O = tuple((A[i] + B[i]) / 2 for i in range(dimension))
    return O, distance(O, A, dimension)


def _plane_equation(subset):
    """Equation du plan passant par 3 points 3D non colineaires."""
    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = subset
    const = x1*y2*z3 + x2*y3*z1 + x3*y1*z2 - x3*y2*z1 - x1*y3*z2 - x2*y1*z3
    c1 = y2*z3 + y3*z1 + y1*z2 - y2*z1 - y3*z2 - y1*z3
    c2 = x1*z3 + x2*z1 + x3*z2 - x3*z1 - x1*z2 - x2*z3
    c3 = x1*y2 + x2*y3 + x3*y1 - x3*y2 - x1*y3 - x2*y1
    return [c1, c2, c3], const


def circum_ball(subset, dimension: int, near_zero: float = 1e-6):
    """
    Cercle/sphere circonscrit(e) a un support de 2 a dimension+1 points.
    Un support intermediaire (3 points en 3D) est contraint au plan des 3
    points. Retourne (None, None) si le support est numeriquement degenere
    (l'appelant doit alors essayer une autre combinaison).
    """
    import numpy as np

    if len(subset) == 2:
        return ball(*subset, dimension)

    coeff, const = [], []
    for P in subset:
        row, c = [], 0.0
        for i in range(dimension):
            row.append(2 * P[i])
            c += P[i] * P[i]
        row.append(1)
        coeff.append(row)
        const.append(c)

    if len(subset) < dimension + 1:
        if dimension != 3 or len(subset) != 3:
            raise ValueError(
                f"circum_ball: support intermediaire non gere hors 3D "
                f"(dimension={dimension}, len(subset)={len(subset)})"
            )
        row, c = _plane_equation(subset)
        row.append(0)   # ne contraint que le centre, pas l'auxiliaire
        coeff.append(row)
        const.append(c)

    u, v = np.array(coeff), np.array(const)
    if abs(np.linalg.det(u)) < near_zero:
        if len(subset) == dimension + 1:
            return None, None   # support complet quasi-degenere : rejete
        # support intermediaire quasi-colineaire : tolere, on resout quand meme
    solved = np.linalg.solve(u, v)
    O = tuple(solved[:dimension])
    return O, distance(O, subset[0], dimension)


# =============================================================== balayage cyclique

def inc_position(sample: list, position: int, step: int, size: int) -> Optional[int]:
    """Avance 'position' de 'step', modulo size, en sautant les entrees None."""
    for _ in range(size):
        position = (position + step) % size
        if sample[position] is not None:
            return position
    return None


def inside(A: tuple, O: tuple, radius_squared: float, dimension: int, precision: float) -> bool:
    return distance(A, O, dimension) <= radius_squared * precision


def point_outside(sample, position, step, size, O, radius_squared, dimension, precision):
    """Recherche cyclique d'un point hors de ball(O, radius_squared)."""
    start = position
    while True:
        E = sample[position]
        if E is not None and not inside(E, O, radius_squared, dimension, precision):
            return E, position
        position = inc_position(sample, position, step, size)
        if position is None or position == start:
            return None, position


def point_outside_track_max(sample, position, step, size, O, radius_squared, dimension, precision):
    """
    Variante de point_outside : en plus du point trouve (ou None), renvoie
    la plus grande distance^2 rencontree parmi les points vivants balayes
    (valable seulement si le cycle est alle a son terme sans rien trouver -
    sert de base a l'invariant local, cf. local_neighborhood_elim).
    """
    start = position
    max_d2 = 0.0
    while True:
        E = sample[position]
        if E is not None:
            d2 = distance(E, O, dimension)
            if d2 > radius_squared * precision:
                return E, position, None
            if d2 > max_d2:
                max_d2 = d2
        position = inc_position(sample, position, step, size)
        if position is None or position == start:
            return None, position, max_d2


# =============================================================== MiniDisk (MEC)

def _find_ball(subset, support_size, dimension, near_zero, precision):
    """Plus petite boule basee sur 'support_size' points de 'subset', englobant le reste."""
    if len(subset) == 2:
        O, r2 = ball(*subset, dimension)
        return O, r2, list(subset)
    support_size = min(support_size, len(subset))

    better = None, float('inf'), []
    for miniset in combinations(subset, support_size):
        O, r2 = circum_ball(miniset, dimension, near_zero) if support_size > 2 else ball(*miniset, dimension)
        if O is None:
            continue
        if support_size == len(subset):
            better = (O, r2, list(miniset))
        else:
            remain = list(set(subset).difference(miniset))
            D, _ = point_outside(remain, 0, 1, len(remain), O, r2, dimension, precision)
            if D is None and r2 < better[1]:
                better = (O, r2, list(miniset))
    return better if better[0] is not None else (None, 0, [])


def minimum_ball(points, subset, position, size, dimension,
                  near_zero: float = 1e-6, max_steps: int = 1000,
                  precision: float = 1 + 1e-12):
    """
    Plus petit cercle/sphere englobant (MEC), incrementalement.
    Retourne (subset, O, rayon^2) ; subset est le support du MEC (2 ou
    dimension+1 points au plus, generalement 2 ou 3 en pratique 2D/3D).
    'precision' donne la marge numerique necessaire pour qu'un point du
    support (exactement sur le bord de son propre cercle circonscrit) ne
    soit pas reclasse "hors du cercle" par erreur d'arrondi (ce qui
    boucterait indefiniment).
    Leve RuntimeError en cas d'anomalie (aucune solution, ou max_steps depasse).
    """
    steps = 0
    while True:
        steps += 1
        if steps > max_steps:
            raise RuntimeError(f"minimum_ball: max_steps ({max_steps}) depasse, subset={subset}")

        support_size = 2
        best = None, float('inf'), []
        while support_size < dimension + 2:
            O, r2, base = _find_ball(subset, support_size, dimension, near_zero, precision)
            if O is not None and r2 < best[1]:
                best = (O, r2, base)
                if support_size == 2:
                    break
            support_size += 1

        O, r2, subset = best
        if O is None:
            raise RuntimeError(f"minimum_ball: aucune solution trouvee pour subset={subset}")

        P, position = point_outside(points, position, 1, size, O, r2, dimension, precision)
        while P is not None and P in subset:
            # frontiere numerique (point du support classe hors cercle par arrondi)
            position = inc_position(points, position, 1, size)
            P, position = point_outside(points, position, 1, size, O, r2, dimension, precision)
        if P is None:
            return subset, O, r2
        subset = subset + [P]


# =============================================================== invariant local

def _volume_ball(r: float, dimension: int) -> float:
    if dimension == 2:
        return math.pi * r * r
    if dimension == 3:
        return (4.0 / 3.0) * math.pi * r ** 3
    raise ValueError(f"_volume_ball: dimension={dimension} non prise en charge (2 ou 3)")


def local_neighborhood_elim(points_mx, size, C, apart, m_squared, R_mec, dimension,
                             seuil_voisinage: float = 1.5) -> int:
    """
    Quand C vient d'etre completement elimine (aucun point vivant restant ne
    depasse le diametre courant AB=sqrt(apart)), et m=sqrt(m_squared) est la
    plus grande distance reelle vue entre C et un point vivant pendant ce
    balayage : tout point F avec dist(C,F) <= eps = (apart-m^2)/(2*AB) verifie
    dist(F,Q) <= eps + m <= AB pour tout Q - eliminable au meme titre que C.
    Declenche seulement si la densite locale attendue depasse seuil_voisinage
    (sinon le second balayage O(size) qu'il impose n'est pas rentable).
    Mute points_mx (marquage None des points elimines). Retourne leur nombre.
    """
    if apart <= 0.0:
        return 0
    AB = math.sqrt(apart)
    eps = (apart - m_squared) / (2 * AB)
    if eps <= 0.0:
        return 0

    r_int = max(0.0, AB - R_mec)
    V_shell = _volume_ball(R_mec, dimension) - _volume_ball(r_int, dimension)
    if V_shell <= 0.0:
        return 0
    n_attendu = (size / V_shell) * _volume_ball(eps, dimension)
    if n_attendu <= seuil_voisinage:
        return 0

    eps_squared = eps * eps
    eliminated = 0
    for i in range(size):
        F = points_mx[i]
        if F is not None and distance(F, C, dimension) <= eps_squared:
            points_mx[i] = None
            eliminated += 1
    return eliminated


# =============================================================== filtre radial

def run_filter(points_mx, size, diameter, mec_center, mec_radius_squared, dimension,
                step: int = 1, precision: float = 1 + 1e-12):
    """
    Elimine (par compaction) tout point a distance <= 2r-R du centre MEC
    (r = demi-diametre courant, R = rayon MEC) : un tel point ne peut plus
    ameliorer le diametre, dans aucun role (inegalite triangulaire).
    Tolere les trous None deja presents dans points_mx (compactes, mais non
    comptes dans elim_inter). Retourne aussi live_before (points non-None
    vus), utile pour juger si un rappel ulterieur est productif.
    """
    size_orig = size
    apart_seed = distance(*diameter, dimension)
    r_seed = math.sqrt(apart_seed) / 2
    R_mec = math.sqrt(mec_radius_squared)
    elim_radius_squared = max(0.0, 2 * r_seed - R_mec) ** 2

    new_size = size
    i = 0
    live_before = 0
    while i < new_size:
        p = points_mx[i]
        if p is not None:
            live_before += 1
        if p is None or inside(p, mec_center, elim_radius_squared, dimension, precision):
            new_size -= 1
            points_mx[i], points_mx[new_size] = points_mx[new_size], points_mx[i]
        else:
            i += 1
    elim_inter = sum(1 for p in points_mx[new_size:size] if p is not None)
    del points_mx[new_size:]
    size = new_size

    if math.gcd(step, size) != 1:
        new_step = 2 * int(step * size / size_orig) - 1
        while math.gcd(new_step, size) != 1:
            new_step += 1
            if new_step >= size:
                new_step = 1
                break
        step = new_step
    position = 2 * step

    return points_mx, size, position, step, elim_inter, live_before


# =============================================================== MaxDist

def run_maxdist(points, seed_diameter, mec_center, mec_radius_squared, dimension,
                 step: int = 1, precision: float = 1 + 1e-12, seuil_voisinage: float = 1.5,
                 radial_recheck_step: int = 8, radial_recheck_period: int = 4,
                 radial_recheck_min_residual_pct: float = 5.0,
                 radial_recheck_min_elim1: float = 2.0):
    """
    Recherche le diametre par amelioration successive a partir de
    'seed_diameter', centre fixe au centre MEC. Combine filtre radial
    (une passe initiale), invariant local (a chaque elimination complete
    d'un point C) et rappel radial periodique (reapplique le filtre radial
    au-dela de radial_recheck_step ameliorations, avec backoff exponentiel
    de l'espacement si un rappel s'avere sterile).
    Retourne (diameter, apart, steps, radial_rappels).
    """
    points_mx = list(points)
    size = len(points_mx)
    size_orig = size
    steps = 1
    position = 2 * step
    diameter = list(seed_diameter)
    R_mec = math.sqrt(mec_radius_squared)

    points_mx, size, position, step, _, _ = run_filter(
        points_mx, size, diameter, mec_center, mec_radius_squared, dimension, step, precision)

    O, radius, apart = None, None, None
    prev_steps = 0
    radial_population_ok = True
    radial_period = radial_recheck_period
    radial_next_check = radial_recheck_step
    radial_rappels = 0

    while True:
        if steps != prev_steps:
            apart = distance(*diameter, dimension)
            O, radius = mec_center, apart / 4
            prev_steps = steps

        position = inc_position(points_mx, position, step, size)
        if position is None:
            break
        C, position = point_outside(points_mx, position, step, size, O, radius, dimension, precision)
        if C is None:
            break

        C_pos = position
        position = inc_position(points_mx, position, step, size)
        if position is None:
            points_mx[C_pos] = None
            break
        D, position, m_squared = point_outside_track_max(
            points_mx, position, step, size, C, apart, dimension, precision)
        if D is None:
            points_mx[C_pos] = None
            local_neighborhood_elim(points_mx, size, C, apart, m_squared, R_mec, dimension, seuil_voisinage)
            continue

        diameter = [C, D]
        steps += 1

        if radial_population_ok and steps >= radial_next_check:
            residual_pct = 100 * size / size_orig
            if residual_pct < radial_recheck_min_residual_pct:
                radial_population_ok = False
            else:
                points_mx, size, position, step, elim_inter_pass, live_before = run_filter(
                    points_mx, size, diameter, mec_center, mec_radius_squared, dimension, step, precision)
                radial_rappels += 1
                pct_this_pass = 100 * elim_inter_pass / live_before if live_before > 0 else 0.0
                radial_period = (radial_period * 2 if pct_this_pass < radial_recheck_min_elim1
                                  else radial_recheck_period)
                radial_next_check = steps + radial_period

    return diameter, apart, steps, radial_rappels


# =============================================================== procedure principale

def max_diameter(points: list, dimension: int, precision: float = 1 + 1e-12, **maxdist_kwargs):
    """
    Diametre exact d'un nuage de points : MiniDisk d'abord (reponse directe
    si son support est une paire), MaxDist ensuite si necessaire (amorce par
    le plus long cote du support MEC). maxdist_kwargs est transmis a
    run_maxdist (seuil_voisinage, radial_recheck_*, ...).
    Retourne (diameter, apart, steps) : diameter = paire de points realisant
    le diametre, apart = distance^2 correspondante, steps = nombre
    d'ameliorations effectuees par MaxDist (0 si MiniDisk a suffi).
    """
    points = list(points)
    if len(points) == 1:
        return (points[0], points[0]), 0.0, 0
    if len(points) == 2:
        return tuple(points), distance(*points, dimension), 0

    subset, O, r_squared = minimum_ball(points, points[:2], 2, len(points), dimension, precision=precision)

    if len(subset) == 2:
        return tuple(subset), distance(*subset, dimension), 0

    seed = max(combinations(subset, 2), key=lambda pq: distance(pq[0], pq[1], dimension))
    diameter, apart, steps, radial_rappels = run_maxdist(
        points, seed, mec_center=O, mec_radius_squared=r_squared, dimension=dimension,
        precision=precision, **maxdist_kwargs)
    return tuple(diameter), apart, steps


# =============================================================== exemple

if __name__ == "__main__":
    import random

    random.seed(0)
    N = 5000
    points = [tuple(random.random() for _ in range(3)) for _ in range(N)]   # cube unite [0,1]^3

    diameter, apart, steps = max_diameter(points, dimension=3)

    print(f"N = {N} points dans un cube [0,1]^3")
    print(f"Diametre = {math.sqrt(apart):.6f}")
    print(f"Paire réalisant le diametre : {diameter}")
    print(f"Ameliorations MaxDist : {steps}")
