from pyscf import gto
from typing import List, Tuple
from util import atoms
from numpy import ndarray


class IntegralCalculator:
    def __init__(self, molecule: List[Tuple[int, float, float, float]]):
        atom_array: List[List[str, Tuple[float, float, float]]] = []
        for i in range(len(molecule)):
            atom_array.append([atoms.symbol[molecule[i][0]],
                              (molecule[i][1], molecule[i][2], molecule[i][3])])
        self.mol = gto.Mole()
        self.mol.atom = atom_array
        self.mol.basis = 'sto-3g'
        self.mol.unit = 'bohr'
        self.mol.build()

        self.S = self.mol.intor('int1e_ovlp')
        self.T = self.mol.intor('int1e_kin')
        self.V = self.mol.intor('int1e_nuc')
        self.TwoEI = self.mol.intor('int2e')

    def getS(self) -> ndarray:
        return self.S

    def getT(self) -> ndarray:
        return self.T

    def getV(self) -> ndarray:
        return self.V

    def getTwoEI(self) -> ndarray:
        return self.TwoEI

    def getE_nuc(self) -> float:
        return self.E_nuc
