from ase.io import read
import numpy as np
from ase.neighborlist import neighbor_list

atoms = read('POSCAR_reconstructed')
z = atoms.positions[:, 2]
max_z = np.max(z)

# Surface Si atoms (top two sub-layers)
top_indices = set([i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'Si' and z[i] > max_z - 2.5])

print(f"Analyzing {len(top_indices)} top surface Si atoms...")

# Identify dimers by distance in the FULL atoms object (to preserve cell/MIC)
i_list, j_list, D_list = neighbor_list('ijD', atoms, 3.0) 

paired = set()
dimers = []
for idx_full, (i, j) in enumerate(zip(i_list, j_list)):
    if i < j and i in top_indices and j in top_indices:
        if i not in paired and j not in paired:
            # Check distance to be sure it's a dimer (2.0 < d < 2.6)
            dist = np.linalg.norm(D_list[idx_full])
            if 2.0 < dist < 2.6:
                dimers.append((i, j, D_list[idx_full]))
                paired.add(i); paired.add(j)

print(f"Found {len(dimers)} dimers.")

tilts = []
for idx1, idx2, dist_vec in dimers:
    p1 = atoms.positions[idx1]
    p2_eff = p1 + dist_vec
    
    # Determine 'left' (lower X or Y)
    if p1[0] < p2_eff[0] - 1e-3 or (abs(p1[0]-p2_eff[0]) < 1e-3 and p1[1] < p2_eff[1] - 1e-3):
        left_idx, right_idx = idx1, idx2
        left_pos, right_pos = p1, p2_eff
    else:
        left_idx, right_idx = idx2, idx1
        left_pos, right_pos = atoms.positions[idx2], atoms.positions[idx2] - dist_vec # reverse vector
    
    # HEIGHT difference (right - left)
    dh = atoms.positions[right_idx][2] - atoms.positions[left_idx][2]
    tilts.append(dh)

unique_tilts = np.unique(np.round(tilts, 4))
print(f"Unique relative heights (right - left) found: {unique_tilts}")

if len(unique_tilts) == 1:
    print(f"SUCCESS: Buckling is uniform across all dimers! (dh={unique_tilts[0]:.4f})")
elif len(unique_tilts) == 0:
    print("FAILURE: No dimers found indices check.")
else:
    print("FAILURE: Buckling orientations are mixed.")
    for i, t in enumerate(tilts):
         print(f"Dimer {i}: tilt={t:.4f}")
