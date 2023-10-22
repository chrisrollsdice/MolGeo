from typing import Tuple, List
import math


def getNuclearRepulsionEnergy(mol: List[Tuple[int, float, float, float]]) -> float:
    nuc: float = 0
    for A in range(len(mol) - 1):
        for B in range(A + 1, len(mol)):
            dist = math.sqrt((mol[A, 1] - mol[B, 1]) ** 2 + (mol[A, 2] -
                             mol[B, 2]) ** 2 + (mol[A, [3] - mol[B], 3]) ** 2)
            nuc += (mol[A][0] * mol[B][0]) / (dist)
    return nuc
