import util.atoms
from typing import Tuple, List


def molecule(input: str) -> List[Tuple[int, float, float, float]]:
    atoms = input.split("; ")
    mol: List[Tuple[int, float, float, float]] = []
    for atom in atoms:
        x = atom.split(" ")
        if util.atoms.atomicNumber(x[0]) == 0 or len(x) != 4:
            continue
        mol.append((util.atoms.atomicNumber(x[0]), float(
            x[1]), float(x[2]), float(x[3])))
    return mol
