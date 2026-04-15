import numpy as np
from ase import Atoms

def find_surface_indices(atoms, side='top', threshold=1.0, species=None):
    """Find indices of atoms at the top or bottom surface based on Z-coordinates."""
    if species:
        indices = np.where(atoms.symbols == species)[0]
    else:
        indices = np.arange(len(atoms))
    
    if len(indices) == 0: return []
    z_coords = atoms.positions[indices, 2]
    
    z_target = np.max(z_coords) if side == 'top' else np.min(z_coords)
    mask = np.abs(z_coords - z_target) < threshold
    return indices[mask]

def check_overlap(atoms, cutoff=1.2, verbose=False):
    """Check for steric overlaps between atoms using a simple distance threshold."""
    from ase.neighborlist import neighbor_list
    i_list, j_list, dists = neighbor_list('ijd', atoms, cutoff)
    if len(i_list) > 0:
        if verbose:
            print(f"Overlap detected: {len(i_list)//2} pairs closer than {cutoff}A")
        return True
    return False

def generate_vsepr_vectors(atoms, idx, neighbor_data=None, num_missing=1, cutoff=2.6):
    """
    Calculate generic dangling bond vectors using VSEPR approximation.
    Distributes num_missing vectors symmetrically around the inverse sum of neighbors.
    """
    from ase.neighborlist import neighbor_list
    if neighbor_data:
        i_list, j_list, D_list = neighbor_data
    else:
        i_list, j_list, D_list = neighbor_list('ijD', atoms, cutoff)
    
    mask = (i_list == idx)
    vectors = D_list[mask]
    
    # Filter to only covalent neighbors
    dists = np.linalg.norm(vectors, axis=1)
    vectors = vectors[(dists > 0.1) & (dists < cutoff)]
    
    if len(vectors) == 0:
        # Fallback: point along Z-axis if no neighbors found
        return [np.array([0., 0., 1.])] * num_missing
        
    # Normalize neighbor vectors
    norm_vecs = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    sum_vec = np.sum(norm_vecs, axis=0)
    
    # Primary direction: away from the geometric center of existing neighbors
    v_target = -sum_vec
    if np.linalg.norm(v_target) < 1e-4:
        v_target = np.array([0., 0., 1.])
    v_target /= np.linalg.norm(v_target)

    if num_missing == 1:
        return [v_target]

    if num_missing == 2 and len(vectors) == 2:
        # Optimized for Tetrahedral/Square-planar geometries (AX2E2)
        # Mirrors the logic for Si(100) surface dimers
        w_unit = v_target
        u = norm_vecs[0] - norm_vecs[1]
        u_norm = np.linalg.norm(u)
        if u_norm > 1e-4:
            u_unit = u / u_norm
            p_unit = np.cross(w_unit, u_unit)
            p_unit /= np.linalg.norm(p_unit)
            
            # tetrahedral coefficients (cos(54.7), sin(54.7))
            v1 = w_unit * 0.577 + p_unit * 0.816
            v2 = w_unit * 0.577 - p_unit * 0.816
            return [v1, v2]

    # General fallback for num_missing > 1: Conical distribution
    results = []
    # Small cone spread for multiple dangling bonds
    theta = np.deg2rad(20.0) 
    
    # Find perpendicular axes
    perp_vec = np.array([1., 0., 0.])
    if abs(np.dot(v_target, perp_vec)) > 0.9:
        perp_vec = np.array([0., 1., 0.])
    
    axis_1 = np.cross(v_target, perp_vec)
    axis_1 /= np.linalg.norm(axis_1)
    axis_2 = np.cross(v_target, axis_1)
    
    for i in range(num_missing):
        phi = 2 * np.pi * i / num_missing
        v = v_target * np.cos(theta) + (axis_1 * np.cos(phi) + axis_2 * np.sin(phi)) * np.sin(theta)
        results.append(v / np.linalg.norm(v))
        
    return results

def get_all_dangling_bonds_general(atoms, valence_map, vector_generator=None, cutoff=3.1, side='top'):
    """
    Identify missing valences for surface atoms using platform-independent logic.
    valence_map: dict {sym: int} or function(symbol) -> int
    vector_generator: function(atoms, idx, neighbor_data, num_missing) -> list of vectors.
                      If None, generate_vsepr_vectors is used.
    """
    from ase.neighborlist import neighbor_list
    surface_indices = find_surface_indices(atoms, side=side, threshold=2.0)
    
    # Use a generous cutoff for neighbor list to share with vector generator
    i_list, j_list, D_list = neighbor_list('ijD', atoms, cutoff)
    neighbor_data = (i_list, j_list, D_list)
    
    if vector_generator is None:
        vector_generator = generate_vsepr_vectors
        
    all_bonds = []
    for idx in surface_indices:
        sym = atoms.symbols[idx]
        target_val = valence_map(sym) if callable(valence_map) else valence_map.get(sym, 0)
        if target_val <= 0: continue

        mask = (i_list == idx)
        dists = np.linalg.norm(D_list[mask], axis=1)
        # Count only true covalent neighbors
        num_n = np.sum((dists > 0.1) & (dists < 2.6))
        num_missing = target_val - num_n
        
        if num_missing > 0:
            # Pass num_missing to the generator
            try:
                vecs = vector_generator(atoms, idx, neighbor_data=neighbor_data, num_missing=num_missing)
            except TypeError:
                # Fallback for generators that don't take num_missing yet
                vecs = vector_generator(atoms, idx, neighbor_data=neighbor_data)
                
            for v in vecs:
                # Basic directional filter: point into vacuum
                if (side == 'top' and v[2] > -0.1) or (side == 'bottom' and v[2] < 0.1):
                    all_bonds.append({'parent': idx, 'vector': v, 'parent_sym': sym})
    return all_bonds

def passivate_surface_coverage_general(atoms, h_coverage, valence_map, vector_generator=None, 
                                       element='H', cutoff=3.1, side='top', verbose=False):
    """Uniformly passivate a surface using a greedy max-min distance algorithm."""
    from ase.geometry import get_distances
    from ase.data import covalent_radii
    
    candidates = get_all_dangling_bonds_general(atoms, valence_map, vector_generator, cutoff, side)
    if not candidates: 
        if verbose: print(f"  [Passivation] No dangling bonds found on {side} surface.")
        return atoms
    
    n_target = int(round(len(candidates) * h_coverage))
    if n_target == 0: return atoms
    
    current_atoms = atoms.copy()
    success = 0
    available = list(candidates)
    
    r_pass = covalent_radii[Atoms(element).numbers[0]]
    
    if verbose: print(f"  [Passivation] Targeting {n_target} {element} sites on {side} surface.")
    
    while success < n_target and available:
        pass_indices = [i for i, sym in enumerate(current_atoms.symbols) if sym == element]
        # Reference positions to maximize distance from (existing passivation + oxygens for steric)
        ref_indices = pass_indices + [i for i, sym in enumerate(current_atoms.symbols) if sym == 'O']
        ref_pos = current_atoms.positions[ref_indices] if ref_indices else []
        
        best_cand_idx = -1
        best_score = -1.0
        
        for i_c, cand in enumerate(available):
            parent_pos = current_atoms.positions[cand['parent']]
            r_parent = covalent_radii[atoms.numbers[cand['parent']]]
            b_len = r_parent + r_pass
            
            h_pos_candidate = parent_pos + cand['vector'] * b_len
            
            if len(ref_pos) == 0:
                score = 100.0
            else:
                dists = get_distances(h_pos_candidate, ref_pos, cell=current_atoms.cell, pbc=current_atoms.pbc)[1]
                score = np.min(dists)
            
            if score > best_score:
                # Overlap check
                _, all_dists_list = get_distances(h_pos_candidate, current_atoms.positions, 
                                                 cell=current_atoms.cell, pbc=current_atoms.pbc)
                all_dists = all_dists_list[0]
                
                mask = np.ones(len(all_dists), dtype=bool)
                mask[cand['parent']] = False
                # Use a combined radius for overlap threshold
                if np.any(all_dists[mask] < 0.8): continue
                
                best_score = score
                best_cand_idx = i_c
        
        if best_cand_idx != -1:
            cand = available.pop(best_cand_idx)
            r_parent = covalent_radii[atoms.numbers[cand['parent']]]
            b_len = r_parent + r_pass
            
            # Empirical refinements if needed (e.g. Si-H, O-H)
            if cand['parent_sym'] == 'Si' and element == 'H': b_len = 1.48
            if cand['parent_sym'] == 'O' and element == 'H': b_len = 0.96
            
            h_pos = current_atoms.positions[cand['parent']] + cand['vector'] * b_len
            current_atoms += Atoms(element, positions=[h_pos])
            current_atoms.wrap()
            success += 1
        else:
            break
            
    if verbose: print(f"  [Passivation] Successfully placed {success}/{n_target} {element} atoms on {side} surface.")
    return current_atoms

def identify_protectors(atoms, config, verbose=False):
    """
    Infers which atoms belong to the protector layer vs the base substrate.
    Based on Z-height and molecular connectivity.
    """
    import numpy as np
    protector_cfg = config.get('protector', {})
    if not protector_cfg.get('enabled', False):
        return np.arange(len(atoms)), np.array([], dtype=int)
        
    heuristic = protector_cfg.get('heuristic', 'graph')
    
    if heuristic == 'tag':
        target_tags = protector_cfg.get('target_tags', [4, 5])
        tags = atoms.get_tags()
        p_mask = np.isin(tags, target_tags)
        s_mask = ~p_mask
        return np.where(s_mask)[0], np.where(p_mask)[0]
        
    elif heuristic in ['z_height', 'graph']:
        # Simple heuristic: trace graph
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix
        from ase.data import covalent_radii
        from ase.geometry import get_distances
        
        n_atoms = len(atoms)
        adj = np.zeros((n_atoms, n_atoms), dtype=int)
        D, d = get_distances(atoms.positions, atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                cutoff = covalent_radii[atoms.numbers[i]] + covalent_radii[atoms.numbers[j]] + 0.3
                if d[i,j] < cutoff and d[i,j] > 0.1:
                    adj[i,j] = 1
                    adj[j,i] = 1
                    
        graph = csr_matrix(adj)
        n_comp, labels = connected_components(csgraph=graph, directed=False)
        
        # Base substrate is usually the largest connected component
        comp_sizes = np.bincount(labels)
        substrate_comp = np.argmax(comp_sizes)
        
        s_mask = labels == substrate_comp
        p_mask = labels != substrate_comp
        
        if verbose:
            print(f"  [Protector Inference] Identified {np.sum(p_mask)} protector atoms and {np.sum(s_mask)} substrate atoms.")
            
        return np.where(s_mask)[0], np.where(p_mask)[0]
        
    return np.arange(len(atoms)), np.array([], dtype=int)
    

class CavityDetector:
    def __init__(self, slab, substrate_indices, protector_indices, grid_res=0.2, verbose=False):
        self.slab = slab
        self.sub_idx = substrate_indices
        self.prot_idx = protector_indices
        self.grid_res = grid_res
        self.verbose = verbose
        
    def find_void_centers(self, top_clearance=4.0):
        import numpy as np
        if len(self.prot_idx) == 0:
            z_max = np.max(self.slab.positions[self.sub_idx, 2]) if len(self.sub_idx) else np.max(self.slab.positions[:, 2])
            return [np.array([
                self.slab.cell[0,0]*0.5, 
                self.slab.cell[1,1]*0.5, 
                z_max + top_clearance
            ])]
            
        from ase.data import vdw_radii
        from scipy.ndimage import distance_transform_edt
        from scipy.ndimage import maximum_filter
        
        cell = self.slab.get_cell()
        lx, ly = cell[0,0], cell[1,1]
        
        z_sub_top = np.max(self.slab.positions[self.sub_idx, 2])
        z_prot_top = np.max(self.slab.positions[self.prot_idx, 2])
        
        if z_prot_top <= z_sub_top:
            return [np.array([lx/2, ly/2, z_sub_top + top_clearance])]
            
        nx = int(np.ceil(lx / self.grid_res))
        ny = int(np.ceil(ly / self.grid_res))
        lz = (z_prot_top + top_clearance) - z_sub_top
        nz = int(np.ceil(lz / self.grid_res))
        
        if nx <= 0 or ny <= 0 or nz <= 0:
            return []
            
        grid = np.ones((nx, ny, nz), dtype=bool)
        
        for idx in self.prot_idx:
            pos = self.slab.positions[idx]
            r = 1.5
            try:
                r = vdw_radii[self.slab.numbers[idx]]
                if np.isnan(r): r = 1.5
            except:
                pass
            
            gx = int((pos[0] % lx) / self.grid_res)
            gy = int((pos[1] % ly) / self.grid_res)
            gz = int((pos[2] - z_sub_top) / self.grid_res)
            
            ir = int(np.ceil((r + 1.2) / self.grid_res)) # 1.2A steric buffer
            x_min, x_max = max(0, gx-ir), min(nx, gx+ir+1)
            y_min, y_max = max(0, gy-ir), min(ny, gy+ir+1)
            z_min, z_max = max(0, gz-ir), min(nz, gz+ir+1)
            grid[x_min:x_max, y_min:y_max, z_min:z_max] = False
            
        dist = distance_transform_edt(grid) * self.grid_res
        
        local_max = maximum_filter(dist, size=3) == dist
        local_max[dist < 0.5] = False
        
        max_coords = np.argwhere(local_max)
        centers = []
        sizes = []
        for c in max_coords:
            x = (c[0] + 0.5) * self.grid_res
            y = (c[1] + 0.5) * self.grid_res
            z = z_sub_top + (c[2] + 0.5) * self.grid_res
            centers.append(np.array([x, y, z]))
            sizes.append(dist[c[0], c[1], c[2]])
            
        centers = [x for _, x in sorted(zip(sizes, centers), key=lambda pair: pair[0], reverse=True)]
        
        if self.verbose:
            print(f"  [CavityDetector] Found {len(centers)} potential void centers inside the protector layer.")
            
        # Cluster centers that are too close to reduce redundancy
        filtered_centers = []
        for c in centers:
            if not filtered_centers:
                filtered_centers.append(c)
            else:
                dists = np.linalg.norm(np.array(filtered_centers) - c, axis=1)
                if np.all(dists > 2.0):
                    filtered_centers.append(c)
            if len(filtered_centers) >= 5: # keep top 5 unique cavities
                break
                
        return filtered_centers
