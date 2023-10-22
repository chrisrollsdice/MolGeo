from energy.hartreefock import HartreeFockEngine
from util import atoms
from typing import Tuple, List, Callable
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, grad
from scipy.optimize import minimize
# Basic Loop:
# evaluate energy
# get gradient
# adjust nuclei
# evaluate energy

def calculateGeometry(mol: List[Tuple[int, float, float, float]], charge: int = 0, basis: str = '3-21g*', unit: str = 'bohr') -> Tuple[float, np.ndarray, bool, int]:
    """
    mol = [(8, 0.000000000000, -0.143225816552, 0.000000000000), 
           (1, 1.638036840407, 1.136548822547, -0.000000000000), 
           (1, -1.638036840407, 1.136548822547, -0.000000000000)]
    """
    #hf = HartreeFockEngine(mol, charge, basis, unit)
    #hf.converge()

    molecule = [(1, 0, 0, 0), (9, 0, 0, 1)]
    r, z = separateRandZ(molecule)
    #print(r)
    #print(z)

    hf_grad = scf.RHF(makeMole(r, z, charge, basis, unit)).apply(grad.RHF)
    hf_grad.verbose = False
    hf_scanner = hf_grad.as_scanner()
    scan = generateScanningFunction(hf_scanner, z, charge, basis, unit)
    f: Callable = extractEnergyFunction(scan)
    g: Callable = extractGradientFunction(scan)


    res = minimize(f, r, method='BFGS', jac=g)
    print(res.x)
    print(res.success)
    print(res.message)

    '''
    hessian = np.eye(gradient.shape[0], gradient.shape[0])
    p = -hessian @ gradient
    # Calculate Step Size using Backtracking Line Search 
    a = calculateStepSize(r, p, e_tot, gradient, scan)
    # Calculate Step and New Position
    s = a * p
    r += s
    #print(gradient)
    #print(a)
    # Calculate the change in Gradient
    prev_gradient = gradient
    e_tot, gradient = scan(r)
    #print(gradient)
    # Update the Inverse Hessian Matrix
    hesssian = approximateInverseHessian(s, gradient - prev_gradient, hessian)
    
    max_iterations = 100
    iter = 1
    convergence_criteria = 1e-5
    for i in range(max_iterations):
        #print(hessian)
        if np.linalg.norm(gradient) < convergence_criteria:
            print("Converged!")
            break
        # Calculate Step Direction
        p = -hessian @ gradient
        # Calculate Step Size using Backtracking Line Search 
        a = calculateStepSize(r, p, e_tot, gradient, scan)
        # Calculate Step and New Position
        s = a * p
       # print(a)s
        r  = r + a * p
        # Calculate the change in Gradient
        prev_gradient = gradient
        e_tot, gradient = scan(r)
        # Update the Inverse Hessian Matrix
        hesssian = approximateInverseHessian(s, gradient - prev_gradient, hessian)
        iter += 1
    print(r)
    '''
    return (0, [], False, 0)

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

def calculateStepSize(r: np.ndarray, p: np.ndarray, val: float, grad: np.ndarray, func: Callable[[np.ndarray], Tuple[float,np.ndarray]], a_0: float = 1.0, rho: float = 0.5, c1: float = 1e-4, c2: float = 0.9):
    alpha = a_0
    r_new = r + alpha * p
    e_new, grad_new = func(r_new)
    while e_new > val + (c1*alpha*p.T@grad) or -p.T@grad_new > -c2*p.T@grad:
        print(alpha)
        alpha *= rho
        r_new = r + alpha * p
        e_new, grad_new = func(r_new)
    return alpha

def approximateInverseHessian(s: np.ndarray, y: np.ndarray, H: np.ndarray) -> np.ndarray:
    d = y.shape[0]
    y = np.reshape(y, (d, 1))
    s = np.reshape(s, (d, 1))
    r = 1/(y.T@s)
    li = (np.eye(d) - (r*((s@(y.T)))))
    ri = (np.eye(d) - (r*((y@(s.T)))))
    return li@H@ri + (r*((s@(s.T))))
    """
    inv_hessian = previous_hessian 
    print(s.T @ y)
    inv_hessian += (s.T @ y + y.T @ previous_hessian @ y) * (s @ s.T)  / ((s.T @ y)**2)
    inv_hessian -= (previous_hessian * y @ s.T + s @ y.T * previous_hessian) / (s.T @ y)
    return inv_hessian
    """