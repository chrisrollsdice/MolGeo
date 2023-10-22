from pyscf import gto
from typing import Tuple, List

# credit to pyquante2 for the list of element symbols
symbol = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F",
          "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]


def atomicNumber(sym: str) -> int:
    """
    Given an element's symbol, returns its atomic number.
    Returns 0 if the element doesn't exist or is larger than Argon.
    """
    if sym == "":
        return 0
    for i in range(len(symbol)):
        if symbol[i] == sym:
            return i
    return 0


def getPYSCFMolecule(molecule: List[Tuple[int, float, float, float]], charge: int, basis: str, unit: str) -> gto.Mole:
    atom_list: List[List[str, Tuple[float, float, float]]] = []
    for i in range(len(molecule)):
        atom_list.append(
            [symbol[molecule[i][0]], (molecule[i][1], molecule[i][2], molecule[i][3])])

    mol = gto.Mole()
    mol.atom = atom_list
    mol.basis = basis
    mol.unit = unit
    mol.charge = charge
    mol.build()

    return mol
