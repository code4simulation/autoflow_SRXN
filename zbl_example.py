import sys, os
import itertools
from ase.io import read
from ase.data import atomic_numbers, atomic_masses
DB_path = 'your_path'

input_elem = sys.argv[1:]
elements = input_elem.copy()
elements.sort(key=lambda x: atomic_numbers[x])

print('pair_coeff         * * e3gnn/parallel ${NMPL} ${POT} ' + f'{" ".join(elements)}\n')

combination = list(itertools.combinations_with_replacement(elements,2))
for idx, combi in enumerate(combination):
    FILE = f"{DB_path}{'-'.join(combi)}/POSCAR"
    if os.path.isfile(FILE):
        tmp = read(FILE)
    else:
        print(f"!!! No {'-'.join(combi)} data !!!")
        continue
    d_eq = tmp.get_distance(0, 1, mic=True)
    id_elem_1 = input_elem.index(combi[0])+1
    id_elem_2 = input_elem.index(combi[1])+1
    if id_elem_2 >= id_elem_1:
        print(f"pair_coeff\t{id_elem_1} {id_elem_2}\tzbl/pair {atomic_numbers[combi[0]]} {atomic_numbers[combi[1]]}\t{0.70*d_eq:.4f} {0.90*d_eq:.4f}")
    else:
        print(f"pair_coeff\t{id_elem_2} {id_elem_1}\tzbl/pair {atomic_numbers[combi[1]]} {atomic_numbers[combi[0]]}\t{0.70*d_eq:.4f} {0.90*d_eq:.4f}")
print('')
for idx, elem in enumerate(elements):
    print(f"mass\t{idx+1} {atomic_masses[atomic_numbers[elem]]:.4f}")
