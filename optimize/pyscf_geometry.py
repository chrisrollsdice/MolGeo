import warnings 
warnings.simplefilter("ignore")

from energy.hartreefock import HartreeFockEngine
from util import atoms
from typing import Tuple, List, Callable
import numpy as np
from pyscf import gto, scf, grad
from scipy.optimize import minimize
# Basic Loop:
# evaluate energy
# get gradient
# adjust nuclei
# evaluate energy

def calculateGeometry(mol: List[Tuple[int, float, float, float]], charge: int = 0, basis: str = '6-31g*', unit: str = 'bohr') -> Tuple[float, np.ndarray, bool, int]:
    """
    mol = [(8, 0.000000000000, -0.143225816552, 0.000000000000), 
           (1, 1.638036840407, 1.136548822547, -0.000000000000), 
           (1, -1.638036840407, 1.136548822547, -0.000000000000)]
    """
    #hf = HartreeFockEngine(mol, charge, basis, unit)
    #hf.converge()

    molecule = [(1, 0, 0, 0), (9, 0, 0, 1)]
    r, z = separateRandZ(molecule)

    hf_grad = scf.RHF(makeMole(r, z, charge, basis, unit)).apply(grad.RHF)
    hf_grad.verbose = False
    hf_scanner = hf_grad.as_scanner()
    scan = generateScanningFunction(hf_scanner, z, charge, basis, unit)
    f: Callable = extractEnergyFunction(scan)
    g: Callable = extractGradientFunction(scan)

    f: Callable = energyFunction(z, charge, basis, unit)
    g: Callable = gradientFunction(z, charge, basis, unit)

    print(g(r))

    max_iter = 200

    res = minimize(f, r, method='BFGS', jac=g, options={'maxiter': max_iter, 'gtol': 5e-4})

    return (f(res.x), res.x, res.success, res.nit)

def separateRandZ(mol: List[Tuple[int, float, float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    r = np.empty(len(mol) * 3)
    z = np.empty(len(mol))
    for i in range(len(mol)):
        z[i] = mol[i][0]
        r[3*i + 0] = mol[i][1]
        r[3*i + 1] = mol[i][2]
        r[3*i + 2] = mol[i][3]
    return (r, z)

def makeMole(r: np.ndarray, z: np.ndarray, charge: str, basis: str, unit: str) -> gto.M:
    atom_list: List[List[str, Tuple[float, float, float]]] = []
    for i in range(z.shape[0]):
        atom_list.append([atoms.symbol[int(z[i])], (r[3*i], r[3*i + 1], r[3*i + 2])])

    mol = gto.Mole()
    mol.atom  = atom_list
    mol.basis = basis
    mol.unit  = unit
    mol.charge = charge
    mol.build()

    return mol

def generateScanningFunction(scanner: Callable[[gto.Mole], Tuple[float, np.ndarray]], z: np.ndarray, charge: int, basis: str, unit: str) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
    def f(r: np.ndarray) -> Tuple[float, np.ndarray]:
        e, g = scanner(makeMole(r, z, charge, basis, unit))
        return (e, g.flatten())
    return f

def extractEnergyFunction(scanning: Callable[[np.ndarray], Tuple[float, np.ndarray]]) -> Callable[[np.ndarray], float]:
    def g(r: np.ndarray) -> float:
        return scanning(r)[0]
    return g

def extractGradientFunction(scanning: Callable[[np.ndarray], Tuple[float, np.ndarray]]) -> Callable[[np.ndarray], float]:
    def g(r: np.ndarray) -> float:
        return scanning(r)[1]
    return g

def evaluateEnergy(x: np.ndarray, z: np.ndarray, charge: str, basis: str, unit: str, density: np.ndarray = None) -> Tuple[float, np.ndarray]:
    mol = []
    for i in range(len(z)):
        mol.append((int(z[i]), x[3*i], x[3*i + 1], x[3*i + 2]))

    hf = HartreeFockEngine(mol, charge, basis, unit)
    hf.converge(density_guess=density)
    if hf.converged == False: print("Warning: hartree-fock failed to converge.")
    return (hf.energy, hf.density_mat)

def energyFunction(z, charge, basis, unit) -> Callable:
    def a(x: np.ndarray) -> float:
        return evaluateEnergy(x, z, charge, basis, unit)[0]
    return a

def evaluateGradient(x: np.ndarray, z: np.ndarray, charge: str, basis: str, unit: str, dx: float = 1e-3) -> np.ndarray:
    energy, density = evaluateEnergy(x, z, charge, basis, unit)
    grad = np.empty(x.shape)
    eye = np.eye(grad.shape[0], grad.shape[0])
    for i in range(grad.shape[0]):
        grad[i] = (evaluateEnergy(x + eye[i]*dx, z, charge, basis, unit, density=density)[0] - energy) / dx
    return grad
    
def gradientFunction(z, charge, basis, unit) -> Callable:
    def b(x: np.ndarray) -> float:
        return evaluateGradient(x, z, charge, basis, unit)
    return b