import sys
from util import parse, fmt
from energy import hartreefock
from optimize import geometry

if __name__ == '__main__':
    args = sys.argv

    # Incorrect Command
    if (len(args) < 3 or len(args) > 5 or (args[1] != 'optimize' and args[1] != 'energy')) and args[1] != 'help':
        print('Error: invalid arguments. Please enter the command in the following format:')
        print('python3 molgeo.py (optimize | energy) \'[molecule]\' [charge] [basis]')
        print('Or type \'python3 molgeo.py help\' for help.')
        quit()
    
    #Help Subommand
    elif args[1] == 'help':
        print('Use the command of the form: ')
        print('python3 molgeo.py (optimize | energy) \'[molecule]\' [charge] [basis]')

        print('Choose \'optimize\' to find the optimal geometry of the molecule, or use \'energy\' to find the hartree-fock energy of the molecule with the given geometry.')
        print('[molecule] is of the form: \'H 1 0 0; O 0 0 0; H -1 0 0\'')
        print('[charge] indicates the charge of the molecule. Defaults to 0.')
        print('[basis] indicates the basis set. Usually is 3-21g*.')
        quit()

    molecule = parse.molecule(args[2])
    charge = 0 if len(args) < 4 else int(args[3])
    basis = '3-21g*' if len(args) < 5 else args[4]
    
    # Energy Subcommand
    if args[1] == 'energy':
        e, density, converged, cycles = hartreefock.calculateHFEnergy(molecule, charge=charge, basis=basis)
        print('Calculated energy: ' + str(e))
        print('\nSuccessfully converged in ' + str(cycles) + ' cycles.' if converged else 'Failed to converge.')
        quit()

    # Optimize Subcommand
    if args[1] == 'optimize':
        e, mol, converged, cycles = geometry.calculateGeometry(molecule, charge=charge, basis=basis)
        print('Minimum energy: ' + str(e))
        print('\nOptimal geometry: ')
        print(fmt.formatMolecule(mol))
        print('Successfully converged in ' + str(cycles) + ' cycles.' if converged else 'Failed to converge.')
        quit()
    