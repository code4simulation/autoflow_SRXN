def reconstruct_2x1_buckled(atoms, bond_length=2.30, buckle=0.7, pattern='checkerboard'):
    """Vector-Agnostic 2x1 reconstruction (Strictly avoids diamond100 assumptions)."""
    indices = find_surface_indices(atoms, 'top')
    if len(indices) == 0: return []
    
    paired, found_dimers = set(), []
    i_list, _ = neighbor_list('ij', atoms, 2.6)
    
    for idx1 in indices:
        if idx1 in paired or np.sum(i_list == idx1) >= 4: continue
        pref_vec = get_natural_pairing_vector(atoms, idx1)
        if pref_vec is None: continue
        
        pos1 = atoms.positions[idx1]
        potential_ids = [i for i in indices if i != idx1 and i not in paired]
        if not potential_ids: continue
        
        D_all, d_all = get_distances(pos1, atoms.positions[potential_ids], cell=atoms.cell, pbc=atoms.pbc)
        D_all, d_all = D_all[0], d_all[0]
        
        best_idx2 = -1
        for sub_id, idx2 in enumerate(potential_ids):
            dist = d_all[sub_id]
            if 2.0 < dist < 4.2:
                if abs(np.dot(D_all[sub_id]/dist, pref_vec)) > 0.8:
                    best_idx2 = idx2
                    best_dist_vec = D_all[sub_id]
                    break
        
        if best_idx2 != -1:
            found_dimers.append({'ids': (idx1, best_idx2), 'dist_vec': best_dist_vec})
            paired.add(idx1); paired.add(best_idx2)

    if not found_dimers: return []

    cell_xy = atoms.cell[:2, :2]
    inv_cell = np.linalg.inv(cell_xy)
    final_organized = []
    
    # Grid assignment for phase parity
    for d in found_dimers:
        p1 = atoms.positions[d['ids'][0]]
        p2_eff = p1 + d['dist_vec']
        d['centroid'] = (p1 + p2_eff) / 2
        
    unique_rows = sorted(list(set(round((d['centroid'][:2] @ inv_cell)[1]*8,1) for d in found_dimers)))
    unique_cols = sorted(list(set(round((d['centroid'][:2] @ inv_cell)[0]*8,1) for d in found_dimers)))
    
    for d in found_dimers:
        r_idx = unique_rows.index(round((d['centroid'][:2] @ inv_cell)[1]*8,1))
        c_idx = unique_cols.index(round((d['centroid'][:2] @ inv_cell)[0]*8,1))
        
        if pattern == 'checkerboard':
            S = (-1)**(r_idx + c_idx)
        elif pattern == 'stripe':
            S = (-1)**c_idx
        else:
            S = 1 # 'uniform' pattern
        
        idx1, idx2 = d['ids']
        # Ensure idx1 is always the one that is 'lower' in coordinate space
        # to prevent random buckling directions within a pattern.
        pos1 = atoms.positions[idx1]
        pos2_eff = pos1 + d['dist_vec']
        
        # Sort based on Cartesian coordinates (X then Y) to be extremely robust
        # This ensures that idx1 is always the "first" atom in a consistent scan direction.
        if d['dist_vec'][0] < -1e-4 or (abs(d['dist_vec'][0]) < 1e-4 and d['dist_vec'][1] < -1e-4):
             idx1, idx2 = idx2, idx1
             curr_dist_vec = -d['dist_vec']
        else:
             curr_dist_vec = d['dist_vec']

        center = atoms.positions[idx1] + curr_dist_vec / 2
        vec = -curr_dist_vec # from idx2 to idx1
        vec /= np.linalg.norm(vec)
        
        d_xy = np.sqrt(bond_length**2 - buckle**2)
        atoms.positions[idx1] = center + vec * (d_xy / 2) + np.array([0, 0, S * buckle / 2])
        atoms.positions[idx2] = center - vec * (d_xy / 2) - np.array([0, 0, S * buckle / 2])
        final_organized.append((idx1, idx2, d['dist_vec'], S))
    return final_organized
