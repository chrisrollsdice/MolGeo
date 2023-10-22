import numpy as np
from energy import ints, nucrepulsion
from scipy import linalg
from typing import Tuple, List
import math


def calculateHFEnergy(mol: List[Tuple[int, float, float, float]]) -> Tuple[float, np.ndarray, bool, int]:
    """
    mol = [(8, 0.000000000000, -0.143225816552, 0.000000000000), 
           (1, 1.638036840407, 1.136548822547, -0.000000000000), 
           (1, -1.638036840407, 1.136548822547, -0.000000000000)]
    """

    E_nuc = nucrepulsion.getNuclearRepulsionEnergy(mol)

    # Step 1: Compute Integrals
    integrals = ints.IntegralCalculator(mol)
    # Step 2: Calculate the core Hamiltonian
    H_core: np.ndarray = integrals.getT() + integrals.getV()
    # Step 3: Construct the Orthogonalizing Matrix
    S_ortho = linalg.fractional_matrix_power(integrals.getS(), -1/2)
    # Step 4: Build the Density Matrix
    # density_guess = calculateDensity(H_core, S_ortho)
    density_guess = np.zeros(H_core.shape)
    D = density_guess

    E_tot = 0

    # First Cycle
    Fock = calculateFockMatrix(H_core, D, integrals.getTwoEI())
    D_new = calculateDensity(Fock, S_ortho)
    E_new = calculateElectronicEnergy(H_core, D, integrals.getTwoEI(), E_nuc)

    deltaE = E_new - E_tot
    deltaD = math.sqrt(np.square(np.subtract(D_new, D)).mean())

    # Cycles
    maxIterations, energyThreshold, densityThreshold = 64, 1e-10, 1e-8
    iterations = 1
    for i in range(maxIterations):
        if deltaE < energyThreshold and deltaD < densityThreshold:
            E_tot = E_new
            D = D_new
            break

        Fock = calculateFockMatrix(H_core, D, integrals.getTwoEI())
        D_new = calculateDensity(Fock, S_ortho)
        E_new = calculateElectronicEnergy(
            H_core, D, integrals.getTwoEI(), E_nuc)

        deltaE = E_new - E_tot
        deltaD = math.sqrt(np.square(np.subtract(D_new, D)).mean())

        iterations += 1
        E_tot = E_new
        D = D_new

    return (E_tot, D, deltaE < energyThreshold and deltaD < densityThreshold, iterations)

# Calculates the density matrix from the fock matrix


def calculateDensity(Fock: np.ndarray, S_ortho: np.ndarray) -> np.ndarray:
    # Transform Fock matrix to orthonormal basis:
    F_prime = (S_ortho.conj().T) @ Fock @ S_ortho
    # Diagonalize Fock matrix
    _, C_prime = linalg.eigh(F_prime)
    # Construct eigenvector matrix
    C = S_ortho @ C_prime
    # Form density matrix
    D = 2 * C[:, :5] @ C[:, :5].T
    return D


def calculateFockMatrix(H_core: np.ndarray, D: np.ndarray, twoEInts: np.ndarray) -> np.ndarray:
    Fock = np.empty(H_core.shape)
    sum = 0
    for u in range(Fock.shape[0]):
        for v in range(Fock.shape[1]):
            sum = H_core[u][v]
            sum += (twoEInts[u, v] * D).sum()
            sum -= 0.5 * (twoEInts[u, :, v] * D).sum()
            Fock[u, v] = sum
    return Fock


def calculateElectronicEnergy(H_core: np.ndarray, D: np.ndarray, twoEInts: np.ndarray, E_nuc: float) -> float:
    F = calculateFockMatrix(H_core, D, twoEInts)
    E_tot = E_nuc
    for u in range(H_core.shape[0]):
        for v in range(H_core.shape[1]):
            E_tot += 0.5 * D[u, v] * (F[u, v] + H_core[u, v])
    return E_tot
