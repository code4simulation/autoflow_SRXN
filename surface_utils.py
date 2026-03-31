import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

def find_surface_indices(atoms, side='top', threshold=0.2):
    """Find indices of atoms at the top or bottom surface based on Z-coordinates."""
    z_coords = atoms.positions[:, 2]
    if side == 'top':
        z_target = np.max(z_coords)
    else:
        z_target = np.min(z_coords)
    return np.where(np.abs(z_coords - z_target) < threshold)[0]

def get_missing_tetrahedral_vectors(atoms, idx, cutoff=2.6, bond_length=1.48):
    """
    Given a Si atom index, identify its missing tetrahedral bond vectors.
    Based on neighbor analysis and tetrahedral geometry (cos(theta) = -1/3).
    """
    # 1. Get neighbors
    i_list, j_list, D_list = neighbor_list('ijD', atoms, cutoff)
    neighbors_D = D_list[i_list == idx]
    
    # Normalize neighbors to unit vectors
    unit_vectors = []
    for d in neighbors_D:
        mag = np.linalg.norm(d)
        if mag > 0.1:
            unit_vectors.append(d / mag)
            
    num_neighbors = len(unit_vectors)
    if num_neighbors >= 4:
        return [] # Fully coordinated
        
    # Geometric solving for tetrahedral symmetry
    # Goal: Find v3, v4 such that dot products with v1, v2 are -1/3
    if num_neighbors == 2:
        v1, v2 = unit_vectors[0], unit_vectors[1]
        
        # Intermediate vectors
        w = v1 + v2 # Bisector direction (scaled)
        u = v1 - v2 # Transverse direction
        if np.linalg.norm(w) < 1e-5: # Opposite bonds? (rare for surface)
            # Pick any perpendicular vector
            p = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
            p = np.cross(v1, p)
            p /= np.linalg.norm(p)
            # Result is 180 degrees apart but tetrahedral requires more
            # In (100) surface this shouldn't happen.
            return [] 

        w_unit = w / np.linalg.norm(w)
        u_unit = u / np.linalg.norm(u)
        p_unit = np.cross(w_unit, u_unit)
        p_unit /= np.linalg.norm(p_unit)
        
        # For an ideal tetrahedron with v1, v2 at angle 109.5:
        # v3 and v4 are in the w-p plane.
        # cos(alpha) = -1/3 relative to each other.
        # k_w for v3/v4 is -1/3? Actually:
        # Let v3 = a*w_unit + b*p_unit, v4 = a*w_unit - b*p_unit
        # |v3|^2 = 1 => a^2 + b^2 = 1
        # v3 . v1 = a * (w_unit.v1) = -1/3
        # w_unit.v1 = |w|/2? No.
        # w = v1+v2. w.v1 = 1 + v1.v2 = 1 - 1/3 = 2/3.
        # |w|^2 = (v1+v2)^2 = 1+1+2(-1/3) = 4/3. |w| = 2/sqrt(3).
        # w_unit . v1 = (2/3) / (2/sqrt(3)) = 1/sqrt(3).
        # So a * (1/sqrt(3)) = -1/3 => a = -sqrt(3)/3 = -1/sqrt(3).
        # a^2 = 1/3. b^2 = 2/3. b = sqrt(2/3).
        
        a_coeff = -1.0 / np.sqrt(3)
        b_coeff = np.sqrt(2.0 / 3.0)
        
        v3 = a_coeff * w_unit + b_coeff * p_unit
        v4 = a_coeff * w_unit - b_coeff * p_unit
        
        # Result is 2 vectors
        return [v3 * bond_length, v4 * bond_length]
        
    return []

def get_natural_pairing_vector(atoms, idx):
    """Determine the lateral direction where dangling bonds point toward a neighbor."""
    vecs = get_missing_tetrahedral_vectors(atoms, idx)
    if len(vecs) == 2:
        # The vector between the two missing bond tips defines the pairing axis
        diff = vecs[0] - vecs[1]
        diff[2] = 0 # Projection onto XY plane
        if np.linalg.norm(diff) > 1e-3:
            return diff / np.linalg.norm(diff)
    return None

def passivate_slab(atoms, species='H', side='bottom', bond_length=1.48):
    """Robust passivation by identifying dangling bonds."""
    indices = find_surface_indices(atoms, side)
    new_h_pos = []
    for idx in indices:
        missing_vecs = get_missing_tetrahedral_vectors(atoms, idx, bond_length=bond_length)
        pos = atoms.positions[idx]
        for vec in missing_vecs:
            # We want to ensure vectors point OUT of the slab
            # For bottom, z should decrease. For top, z should increase.
            if (side == 'bottom' and vec[2] < 0) or (side == 'top' and vec[2] > 0):
                new_h_pos.append(pos + vec)
                
    if new_h_pos:
        atoms += Atoms(species * len(new_h_pos), positions=new_h_pos)
    return atoms

def reconstruct_2x1_buckled(atoms, bond_length=2.30, buckle=0.7):
    """
    Automatic 2x1 reconstruction for top surface.
    Identifies pairs based on proximity and missing neighbors.
    """
    indices = find_surface_indices(atoms, 'top')
    # Use neighbor list to find coordination
    i_list, j_list = neighbor_list('ij', atoms, 2.6)
    
    # Filter for undercoordinated Si (Coord < 4)
    surface_si = []
    for idx in indices:
        if np.sum(i_list == idx) < 4:
            surface_si.append(idx)
            
    # Pairing algorithm (Heuristic: Pair closest available neighbors)
    paired = set()
    dimer_pairs = []
    
    for idx1 in surface_si:
        if idx1 in paired: continue
        
        # Determine the "desired" pairing axis from dangling bonds
        pref_vec = get_natural_pairing_vector(atoms, idx1)
        if pref_vec is None: continue
        
        pos1 = atoms.positions[idx1]
        best_idx2 = -1
        best_dist = 10.0
        
        for idx2 in surface_si:
            if idx2 == idx1 or idx2 in paired: continue
            pos2 = atoms.positions[idx2]
            dist_vec = pos2 - pos1
            dist = np.linalg.norm(dist_vec)
            
            # Check if this neighbor is along the preferred pairing axis
            # (Dot product should be close to 1 or -1)
            dist_unit = dist_vec / dist
            alignment = abs(np.dot(dist_unit, pref_vec))
            
            if dist < 4.5 and alignment > 0.8: # Must be close and aligned
                if dist < best_dist:
                    best_dist = dist
                    best_idx2 = idx2
        
        if best_idx2 != -1:
            dimer_pairs.append((idx1, best_idx2))
            paired.add(idx1)
            paired.add(best_idx2)
            
    # Apply Buckling and Shift
    d_xy = np.sqrt(bond_length**2 - buckle**2)
    for idx1, idx2 in dimer_pairs:
        p1, p2 = atoms.positions[idx1], atoms.positions[idx2]
        center = (p1 + p2) / 2
        vec = (p1 - p2) / np.linalg.norm(p1 - p2)
        
        # Apply Buckle (idx1 up, idx2 down)
        atoms.positions[idx1] = center + vec * (d_xy / 2) + np.array([0, 0, buckle / 2])
        atoms.positions[idx2] = center - vec * (d_xy / 2) - np.array([0, 0, buckle / 2])
        
    print(f"Applied 2x1 reconstruction to {len(dimer_pairs)} pairs.")
    return dimer_pairs
