from scipy.optimize import minimize
from pyscf import gto
import numpy as np
from typing import Tuple, List, Callable
from util import atoms
from energy.hartreefock import HartreeFockEngine
import warnings
warnings.simplefilter("ignore")


def calculateGeometry(mol: List[Tuple[int, float, float, float]], charge: int = 0, basis: str = '6-31g*', unit: str = 'bohr') -> Tuple[float, List[Tuple[int, float, float, float]], bool, int]:

    x, z = separateCoordsAndCharge(mol)

    f: Callable = energyFunction(z, charge, basis, unit)
    g: Callable = gradientFunction(z, charge, basis, unit)

    max_iter = 50 * z.shape[0]

    res = minimize(f, x, method='BFGS', jac=g, options={
                   'maxiter': max_iter, 'gtol': 5e-4})

    return (f(res.x), joinCoordsAndCharge(res.x, z), res.success, res.nit)


def separateCoordsAndCharge(mol: List[Tuple[int, float, float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.empty(len(mol) * 3)
    z = np.empty(len(mol))
    for i in range(len(mol)):
        z[i] = mol[i][0]
        x[3*i + 0] = mol[i][1]
        x[3*i + 1] = mol[i][2]
        x[3*i + 2] = mol[i][3]
    return (x, z)


def joinCoordsAndCharge(x: np.ndarray, z: np.ndarray) -> List[Tuple[int, float, float, float]]:
    mol = []
    for i in range(len(z)):
        mol.append((int(z[i]), x[3*i], x[3*i + 1], x[3*i + 2]))
    return mol


def makeMole(x: np.ndarray, z: np.ndarray, charge: str, basis: str, unit: str) -> gto.M:
    atom_list: List[List[str, Tuple[float, float, float]]] = []
    for i in range(z.shape[0]):
        atom_list.append(
            [atoms.symbol[int(z[i])], (x[3*i], x[3*i + 1], x[3*i + 2])])

    mol = gto.Mole()
    mol.atom = atom_list
    mol.basis = basis
    mol.unit = unit
    mol.charge = charge
    mol.build()

    return mol


def evaluateEnergy(x: np.ndarray, z: np.ndarray, charge: str, basis: str, unit: str, density: np.ndarray = None) -> Tuple[float, np.ndarray]:
    mol = []
    for i in range(len(z)):
        mol.append((int(z[i]), x[3*i], x[3*i + 1], x[3*i + 2]))

    hf = HartreeFockEngine(mol, charge, basis, unit)
    hf.converge(density_guess=density,
                density_threshold=1e-4, max_iterations=128)
    if hf.converged == False:
        print("Warning: hartree-fock failed to converge.")
    return (hf.energy, hf.density_mat)


def energyFunction(z, charge, basis, unit) -> Callable:
    d_last = [None]
    def a(x: np.ndarray) -> float:
        e, d = evaluateEnergy(x, z, charge, basis, unit, density=d_last[0])
        d_last[0] = d
        return e
    return a


def evaluateGradient(x: np.ndarray, z: np.ndarray, charge: str, basis: str, unit: str, dx: float = 1e-3, last_density: np.ndarray = None) -> np.ndarray:
    energy, density = evaluateEnergy(x, z, charge, basis, unit)
    grad = np.empty(x.shape)
    eye = np.eye(grad.shape[0], grad.shape[0])
    for i in range(grad.shape[0]):
        grad[i] = (evaluateEnergy(x + eye[i]*dx, z, charge,
                   basis, unit, density=density)[0] - energy) / dx
    return (grad, density)


def gradientFunction(z, charge, basis, unit) -> Callable:
    def b(x: np.ndarray) -> float:
        g, _ = evaluateGradient(x, z, charge, basis, unit)
        return g
    return b
