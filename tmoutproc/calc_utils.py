"""
Methods usefull for calculations using turbomole output
"""
__docformat__ = "google"
#__all__
import numpy as np
import fnmatch
from . import constants
from . import io as io

__ang2bohr__ = 1.88973

def lagrangian(K, dimension=3):
    '''
    Lagrange-multiplier formalism to adjust force-constant matrix K in a way to fulfill the acoustic sum rule ASR. 
    The method is implemented according to:
    N. Mingo, D. A. Stewart, D. A. Broido and D. Srivastava, "Phonon transmission through defects in carbon nanotubes from first principles", 
    PHYSICAL REVIEW B 77, 033418 2008
    DOI: 10.1103/PhysRevB.77.033418

    Args:
        param1 (np.ndarray) : Force constant matrix
        param2 (int) : Number of dimensions

    Returns:
        np.ndarray
    '''
    
    def extract_displacement_modes(K, dimension):

        eigenvalues, eigenvectors = np.linalg.eigh(K)
        sorted_indices = np.argsort(eigenvalues)
        
        if dimension == 3:
            R = eigenvectors[:, sorted_indices[:6]].T
        elif dimension == 2:
            R = eigenvectors[:, sorted_indices[:3]].T
        elif dimension == 1:
            R = eigenvectors[:, sorted_indices[:1]].T 

        return R

    def calculate_B(K, R):
        K_sq = K * K
        term1_sum = 0.25 * np.einsum('ik, mk, nk -> inm', K_sq, R, R, optimize=True)
        n_rows = K.shape[0]
        n_constraints = R.shape[0]
        B_term1 = np.zeros((n_constraints, n_constraints, n_rows, n_rows))

        for i in range(n_rows):
            B_term1[:, :, i, i] = term1_sum[i, :, :]
    
        term2 = 0.25 * np.einsum('mi, nj, ij -> nmij', R, R, K_sq, optimize=True)
        B_term2 = term2.transpose(1, 0, 2, 3)
        
        return B_term1 + B_term2

    def calculate_a(K, R):
        a = -np.einsum('ij, mj -> im', K, R, optimize=True)
        
        return a

    def solve_lagrange_multipliers(B, a):
        n_constraints, n_rows = a.shape
        B_full = B.transpose(2, 0, 3, 1).reshape(n_rows * n_constraints, n_rows * n_constraints)
        a_flat_correct = a.reshape((n_constraints * n_rows,))
        
        try:
            lambda_full = np.linalg.solve(B_full, a_flat_correct)
        except np.linalg.LinAlgError:
            lambda_full = np.linalg.lstsq(B_full, a_flat_correct, rcond=None)[0]
    
        lambda_reshaped = lambda_full.reshape(n_constraints, n_rows)

        return lambda_reshaped

    def calculate_D(K, R, lambda_):
        term1_sum = np.einsum('mi, mj -> ij', lambda_, R, optimize=True)
        term2_sum = term1_sum.T
        sum_of_sums = term1_sum + term2_sum
        K_sq = K * K
        D = 0.25 * K_sq * sum_of_sums
        
        return D

    R = extract_displacement_modes(K, dimension)
    B = calculate_B(K, R)
    a = calculate_a(K, R)
    lambda_ = solve_lagrange_multipliers(B, a)
    D = calculate_D(K, R, lambda_)
    
    return K + D

def create_dynamical_matrix(filename_hessian, filename_coord, t2SI=False, dimensions=3):
    """
    Creates dynamical matrix by mass weighting hessian. Input units are hartree/bohr**2.1 Default output in turbomole format (hartree/(bohr**2*u))

    Args:
        param1 (String) : Filename to hessian
        param2 (String) : Filename to coord file (xyz or turbomole format)
        param3 (boolean) : convert ouptput from turbomole (hartree/(bohr**2*u)) in SI units
        param3 (int) : number of dimensions

    Returns:
        np.ndarray
    """
    # read atoms from file
    atoms = list()
    # check if xyz or turbomoleformat is parsed
    if (fnmatch.fnmatch(filename_coord, '*.xyz')):
        datContent = io.read_xyz_file(filename_coord)
        datContent = datContent[0,:]
        atoms.extend(datContent)
    else:
        datContent = io.read_coord_file(filename_coord)
        atoms = datContent[3,:]
    # determine atom masses
    masses = list()
    for i in range(0, len(atoms)):
        if (t2SI == True):
            masses.append(atom_weight(atoms[i], u2kg=True))
        else:
            masses.append(atom_weight(atoms[i]))

    # create dynamical matrix by mass weighting hessian
    hessian = io.read_hessian(filename_hessian, len(atoms), dimensions)

    dynamical_matrix = np.zeros((len(atoms) * dimensions, len(atoms) * dimensions))
    for i in range(0, len(atoms) * dimensions):
        if (t2SI == True):
            # har/bohr**2 -> J/m**2
            hessian[i, i] = hessian[i, i] * constants.HAR2JOULE/constants.BOHR2METER**2

        dynamical_matrix[i, i] = (1. / np.sqrt(masses[int(i / dimensions)] * masses[int(i / dimensions)])) * hessian[i, i]
        for j in range(0, i):
            # har/bohr**2 -> J/m**2
            if (t2SI == True):
                hessian[i, j] = hessian[i, j] * constants.HAR2JOULE/constants.BOHR2METER**2
            dynamical_matrix[i, j] = (1. / np.sqrt(masses[int(i / dimensions)] * masses[int(j / dimensions)])) * \
                                     hessian[i, j]
            dynamical_matrix[j, i] = dynamical_matrix[i, j]
    return dynamical_matrix


def find_c_range_atom(atom, number, coord, basis_set="dev-SV(P)"):
    """
    finds atoms range of expansion coef in mos file. number gives which atoms should be found e.g. two sulfur : first number =1; second = 2

    Args:
        param1 (String): atom type
        param2 (int): number of atom type atom
        param3 (String): coord file (from io.read_coord_file())
        param4 (String): basis_set="dev-SV(P)"


    Returns:
        tuple (index_start, index_end)
    """

    def atom_type_to_number_coeff(atom):
        """
        returns ordinal number of atom

        Args:
            param1 (String): atom type

        Returns:
            ordinal number (int)
            """
        #TODO: NO hard coded version
        if (atom == "c"):
            return 14
        elif (atom == "h"):
            return 2
        elif (atom == "au"):
            return 25
        elif (atom == "s"):
            return 18
        elif (atom == "n"):
            return 14
        elif (atom == "zn"):
            return 24
        elif (atom == "o"):
            return 14
        elif (atom == "fe"):
            return 24
        elif (atom == "f"):
            return 14
        elif (atom == "cl"):
            return 18
        elif (atom == "br"):
            return 32
        elif (atom == "i"):
            return 13

        else:
            raise ValueError('Sorry. This feature is not implemented for following atom: ' + atom)


    if (basis_set == "dev-SV(P)"):
        number_found_atoms_of_type = 0
        number_expansion_coeff = 0
        for i in range(0, coord.shape[1]):
            current_atom = coord[3,i]

            if (current_atom == atom):
                number_found_atoms_of_type += 1
                if (number_found_atoms_of_type == number):
                    return (number_expansion_coeff, number_expansion_coeff + atom_type_to_number_coeff(current_atom))
                number_expansion_coeff += atom_type_to_number_coeff(current_atom)
            else:
                number_expansion_coeff += atom_type_to_number_coeff(current_atom)
        raise ValueError(f'Sorry. Could not find the atom {current_atom} or {atom}')


    else:
        raise ValueError('Sorry. This feature is not implemented for following basis_set: ' + basis_set)


def atom_weight(atom, u2kg=False):
    """
    Return mass of atom in kg or au (au is default)

    Args:
        param1 (String) : Atom type
        param2 (boolean) : convert to kg

    Returns:
        float
    """
    atom = atom.lower()

    u = constants.U2KG
    if (u2kg == False):
        u = 1
    if (atom == "test"):
        return 1.0
    try:
        dict_entry = constants.ATOM_DICT_SYM[atom]
        atom_weight = dict_entry[2]
    except KeyError as e:
        error_text = (f"Element {atom} cannot be found. Raising exception {e}")
        raise ValueError(error_text)

    return atom_weight * u


def atom_type_to_atomic_number(atom_type):
    """
    Returns atomic_number of atom given in atom_type

    Args:
        param1 (String): atom type

    Returns:
        ordinal number (int)
    """
    try:
        dict_entry = constants.ATOM_DICT_SYM[atom_type.lower()]
        Z = dict_entry[0]
    except KeyError as e:
        error_text = (f"Element {atom_type} cannot be found. Raising exception {e}")
        raise ValueError(error_text)
    return Z


def determine_n_orbitals(coord_path, basis_set="dev-SV(P)"):
    """
    Determines number of orbitals for calculation using coord file (turbomole format, xyz support is planned) and basis set.

    Args:
        coord_path (String): path to coord file (turbomole coord file)
        basis_set (String): basis set

    Returns:
        n_orbitals (int): Number of orbitals

    """
    if (basis_set != "dev-SV(P)"):
        raise ValueError("Only support for dev-SV(P) Basis set right now.")

    coord = io.read_coord_file(coord_path)
    atom_dict = {}
    ranges = list()
    for i, atom in enumerate(coord[3,:]):
        try:
            #atom_dict[item[3]] += 1
            atom_dict[atom] += 1
        except KeyError:
            #atom_dict[item[3]] = 1
            atom_dict[atom] = 1

        start, end = find_c_range_atom(atom, atom_dict[atom], coord)
        ranges.append((start, end, atom))
    n_orbitals = ranges[-1][1]

    return n_orbitals


def get_norb_from_config(config):
    """
    Get number of orbitals from turbomole config string (for example [6s3p2d])

    Args:
        config (String): configuration string

    Returns:
        Norb (int): Number of orbitals
    """
    orbital_types = ['s', 'p', 'd', 'f']
    orbital_degeneracy = [1, 3, 5, 7]
    number_of_orbital_types = np.zeros(len(orbital_types))

    for i in range(0, len(orbital_types)):
        ns, config = config.split(orbital_types[i])
        ns = int(ns)
        number_of_orbital_types[i] = ns
        if config == '':
            break
    if (config != ''):
        raise ValueError(f"There is something wrong with this configuration")

    Norb = np.sum(number_of_orbital_types * orbital_degeneracy)
    return int(Norb)




