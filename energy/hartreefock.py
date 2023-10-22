from pyscf.gto import Mole
from scipy.linalg import eigh, fractional_matrix_power
from util import atoms
from typing import Tuple, List
import math
import numpy as np


class HartreeFockEngine():
    def __init__(self, molecule: List[Tuple[int, float, float, float]], charge: int, basis: str, unit: str) -> None:
        # PYSCF Molecule
        self.mol: Mole = atoms.getPYSCFMolecule(molecule, charge, basis, unit)
        self.n_electrons: int = self.mol.tot_electrons()

        # Energies and Integrals
        self.nuc_energy = HartreeFockEngine.calculateNuclearRepulsionEnergy(
            molecule)
        self.overlap_integrals = self.mol.intor('int1e_ovlp')
        self.kinetic_integrals = self.mol.intor('int1e_kin')
        self.nuclear_integrals = self.mol.intor('int1e_nuc')
        self.two_e_integrals = self.mol.intor('int2e')

        # Hamiltonian
        self.core_hamiltonian = self.kinetic_integrals + self.nuclear_integrals

        # Orthogonalizing Matrix
        self.ortho_mat = fractional_matrix_power(self.overlap_integrals, -1/2)

        # Program Outputs
        self.energy: float = 0
        self.density_mat: np.ndarray = np.zeros((self.mol.nao, self.mol.nao))
        self.cycles: int = 0
        self.converged: bool = False

    def calculateDensity(self, fock_mat: np.ndarray) -> np.ndarray:
        # Transform Fock matrix to orthonormal basis:
        fock_ortho = (self.ortho_mat.conj().T) @ fock_mat @ self.ortho_mat
        # Diagonalize Fock matrix
        _, coeff_ortho = eigh(fock_ortho)
        # Construct eigenvector matrix to get the orbital coefficients
        coeff_mat = self.ortho_mat @ coeff_ortho
        # Form density matrix
        density_mat = 2 * \
            coeff_mat[:, :(self.n_electrons//2)] @ coeff_mat[:,
                                                             :(self.n_electrons//2)].T
        return density_mat

    def calculateFock(self, density_mat: np.ndarray) -> np.ndarray:
        fock_mat = np.empty((self.mol.nao, self.mol.nao))
        for u in range(fock_mat.shape[0]):
            for v in range(fock_mat.shape[1]):
                # F_uv = H_uv + Coulomb Integral - 1/2 * Exchange Integral
                coulomb = (self.two_e_integrals[u, v] * density_mat).sum()
                exchange = (self.two_e_integrals[u, :, v] * density_mat).sum()
                fock_mat[u, v] = self.core_hamiltonian[u, v] + \
                    coulomb - (0.5 * exchange)
        return fock_mat

    def calculateEnergy(self, density_mat: np.ndarray) -> float:
        fock_mat = self.calculateFock(density_mat)
        # Energy = Sum over uv: D_uv (H_uv + F_uv) + E_nuc
        total_energy = self.nuc_energy + \
            (0.5 * density_mat * (fock_mat + self.core_hamiltonian)).sum()
        return total_energy

    def calculateNuclearRepulsionEnergy(molecule: List[Tuple[int, float, float, float]]) -> float:
        nuc: float = 0
        for A in range(len(molecule) - 1):
            for B in range(A + 1, len(molecule)):
                dx = (molecule[A][1] - molecule[B][1])
                dy = (molecule[A][2] - molecule[B][2])
                dz = (molecule[A][3] - molecule[B][3])
                nuc += (molecule[A][0] * molecule[B][0]) / \
                    math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return nuc

    def converge(self, density_guess: np.ndarray = None, max_iterations: int = 64, energy_threshold: float = 1e-10, density_threshold: float = 1e-6) -> float:
        # Initial density is the guess, initial energy is 0
        density_mat = np.zeros((self.mol.nao, self.mol.nao)
                               ) if density_guess is None else density_guess
        total_energy = 0

        # First Cycle: calculate new fock matrix from density, new density from fock matrix
        fock_mat = self.calculateFock(density_mat)
        new_density = self.calculateDensity(fock_mat)
        new_energy = self.calculateEnergy(new_density)

        # Differences: calculate the change in energy and density
        delta_energy = new_energy - total_energy
        delta_density = math.sqrt(
            np.square(np.subtract(new_density, density_mat)).mean())

        # Cycles
        iterations = 1
        for i in range(max_iterations):
            # Check for convergence (both change in energy and density below thresholds)
            if delta_energy < energy_threshold and delta_density < density_threshold:
                total_energy = new_energy
                density_mat = new_density
                break
            # Hasn't converged; generate new fock matrix and new density matrix
            fock_mat = self.calculateFock(new_density)
            new_density = self.calculateDensity(fock_mat)
            new_energy = self.calculateEnergy(new_density)
            # Diffs
            delta_energy = new_energy - total_energy
            delta_density = math.sqrt(
                np.square(np.subtract(new_density, density_mat).real).mean())
            # Keep track of iterations
            iterations += 1
            # Update energy and density
            total_energy = new_energy
            density_mat = new_density
        # store results
        self.energy = total_energy
        self.density_mat = density_mat
        self.cycles = iterations
        self.converged = (
            delta_energy < energy_threshold and delta_density < density_threshold)

        return self.energy


def calculateHFEnergy(mol: List[Tuple[int, float, float, float]], charge: int = 0, basis: str = 'sto-3g', unit: str = 'bohr') -> Tuple[float, np.ndarray, bool, int]:
    """
    mol = [(8, 0.000000000000, -0.143225816552, 0.000000000000), 
           (1, 1.638036840407, 1.136548822547, -0.000000000000), 
           (1, -1.638036840407, 1.136548822547, -0.000000000000)]
    """
    hf = HartreeFockEngine(mol, charge, basis, unit)
    hf.converge(density_threshold=1e-8, max_iterations=50)
    return (hf.energy, hf.density_mat, hf.converged, hf.cycles)
