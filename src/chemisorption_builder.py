import numpy as np
from ase import Atoms
from ads_workflow_mgr import AdsorptionWorkflowManager
def analyze_surface_reactivity(surface, config, verbose=True):
    """
    Analyzes the surface geometrically to find generically reactive sites.
    Uses 'ideal_coordination' from config to detect undercoordinated atoms (dangling bonds).
    Returns:
        pairs: list of ((idx1, db_info1), (idx2, db_info2)) pairs within max_pair_dist.
        single_sites: list of (idx, db_info)
    """
    from ase.neighborlist import neighbor_list
    import numpy as np
    from ase.data import covalent_radii
    
    ideal_coord = config.get('ideal_coordination', {})
    max_pair_dist = config.get('settings', {}).get('max_pair_dist', 5.0)
    
    # Identify Dangling Bonds (Undercoordinated sites)
    # Using ijD to get displacement vectors for VSEPR
    i_list, j_list, D_list = neighbor_list('ijD', surface, cutoff=3.0) 
    d_list = np.linalg.norm(D_list, axis=1)
    
    dangling_sites = []
    
    # We only care about top surface atoms
    z_max = max(surface.positions[:, 2])
    surface_mask = surface.positions[:, 2] > (z_max - 2.5)
    
    for idx in range(len(surface)):
        if not surface_mask[idx]: continue
        sym = surface.symbols[idx]
        if sym not in ideal_coord: continue
        
        # Count actual bonds
        neighbors = []
        for n_i, n_j, dist in zip(i_list, j_list, d_list):
            if n_i == idx:
                cutoff_val = covalent_radii[surface.numbers[n_i]] + covalent_radii[surface.numbers[n_j]] + 0.3
                if dist < cutoff_val and dist > 0.1:
                    neighbors.append(n_j)
                    
        actual_coord = len(neighbors)
        expected = ideal_coord[sym]
        
        if actual_coord < expected:
            from surface_utils import generate_vsepr_vectors
            # Calculate dangling vectors using centralized utility
            vecs = generate_vsepr_vectors(surface, idx, neighbor_data=(i_list, j_list, D_list))
            db_vec = vecs[0]
                
            dangling_sites.append({
                'index': idx,
                'sym': sym,
                'pos': surface.positions[idx],
                'db_vector': db_vec,
                'missing_bonds': expected - actual_coord
            })
            
    if verbose:
        print(f"  [Generic Reactivity] Identified {len(dangling_sites)} undercoordinated surface sites.")
        
    results = {'single': dangling_sites, 'pairs': []}
    
    # Analyze Symmetry to reduce pair redundancies
    import spglib
    lattice = surface.get_cell()
    pos = surface.get_scaled_positions()
    nums = surface.get_atomic_numbers()
    symprec = config.get('settings', {}).get('symprec', 0.2)
    
    equiv_atoms = np.arange(len(surface))
    for prec in [symprec, 0.5]:
        try:
            dataset = spglib.get_symmetry_dataset((lattice, pos, nums), symprec=prec)
            if dataset:
                equiv_atoms = dataset.equivalent_atoms if hasattr(dataset, 'equivalent_atoms') else dataset['equivalent_atoms']
                if len(np.unique(equiv_atoms)) < len(surface) or prec == 0.5:
                    break
        except Exception:
            pass
            
    from itertools import combinations
    unique_pairs = {}
    pair_count = 0
    
    for s1, s2 in combinations(dangling_sites, 2):
        dist = np.linalg.norm(s1['pos'] - s2['pos'])
        if dist <= max_pair_dist:
            pair_count += 1
            # Pair signature: sorted tuple of symmetry classes + rounded distance
            c1 = equiv_atoms[s1['index']]
            c2 = equiv_atoms[s2['index']]
            sig = tuple(sorted([c1, c2])) + (round(dist, 1),)
            
            if sig not in unique_pairs:
                unique_pairs[sig] = (s1, s2)
            
    results['pairs'] = list(unique_pairs.values())
    
    if verbose:
        print(f"  [Generic Reactivity] Formed {pair_count} active site pairs -> Symmetry reduced to {len(results['pairs'])} unique pairs.")
        
    return results


def analyze_molecule_ligands(molecule, center_target='Si', verbose=True):
    """
    Algorithmically fragments the precursor molecule to identify reactive ligands.
    Uses AdsorptionWorkflowManager implicitly for the heavy lifting.
    """
    # Create a temporary manager to use its fragmentation logic
    mgr = AdsorptionWorkflowManager(molecule, verbose=verbose)
    c_idx, ligands = mgr.discover_ligands(molecule, center_target=center_target, verbose=verbose)
    return c_idx, ligands


def build_chemisorption_structures(molecule, center_target='Si', surface=None, rot_steps=8, config=None, verbose=True):
    """
    Entry point for algorithmic chemisorption generation based on input molecule and surface.
    Identifies valid mechanisms based on available surface sites.
    """
    if verbose:
        print("\n--- Starting Algorithmic Chemisorption Routing ---")
        
    if config is None: config = {}
    sites = analyze_surface_reactivity(surface, config, verbose=verbose)
    c_idx, ligands = analyze_molecule_ligands(molecule, center_target=center_target, verbose=verbose)
    
    candidates = []
    
    if not ligands:
        if verbose: print("  [Warning] No detachable ligands found. Aborting chemisorption.")
        return candidates
        
    # We instantiate a manager scoped to the current surface for coordinate placement/overlap tests
    mgr = AdsorptionWorkflowManager(surface, verbose=verbose)
    
    # Generic Cohesive Dissociation on active site pairs
    if sites.get('pairs'):
        if verbose: print("  -> Routing to Generic Dissociative Chemisorption on Pairs...")
        d_cands = _execute_generic_dissociation(mgr, molecule, c_idx, ligands, sites['pairs'], rot_steps)
        candidates.extend(d_cands)
        
    # We can also add Single-site generic additions if do_chemisorption includes it.
        
    if verbose:
        print(f"--- Finished Chemisorption Builder. Total Generated: {len(candidates)} ---\n")
        
    return candidates


def _execute_generic_single_site(mgr, molecule, c_idx, ligands, sites, rot_steps):
    """ Internal subroutine to execute Generic Single Site Addition/Exchange """
    candidates = []
    stats = {'overlap': 0, 'deduplicated': 0}
    seen_formulas = set()
    
    for l_info in ligands:
        formula = l_info.get('formula', 'Unknown')
        if formula in seen_formulas: 
            stats['deduplicated'] += 1
            continue
        seen_formulas.add(formula)
        
        indices_b = l_info['indices']
        frag_b = molecule[indices_b]
        binding_idx_b = indices_b.index(l_info['binding_atoms'][0])
        
        indices_a = list(set(range(len(molecule))) - set(indices_b))
        frag_a = molecule[indices_a]
        binding_idx_a = indices_a.index(c_idx)
        
        for s in sites:
            si_pos = s['pos']
            h_vec_norm = s['db_vector']
            
            for angle in np.linspace(0, 360, rot_steps, endpoint=False):
                p_a = mgr._place_at_dangling_bond(frag_a, binding_idx_a, l_info['bond_vec'], 
                                                   si_pos, h_vec_norm, 2.35, rot_angle=angle)
                
                p_b = mgr._form_byproduct(frag_b, binding_idx_b, -l_info['bond_vec'])
                z_clearance = np.max(mgr.slab.positions[:, 2]) + 4.0
                p_b.translate([si_pos[0], si_pos[1], z_clearance] - p_b.positions[0])
                
                # Here we DO NOT drop any atom because this is addition. If exchange is needed, we would drop the H.
                combined = mgr.slab.copy()
                
                for a in p_a: a.tag = 2
                combined += p_a
                for a in p_b: a.tag = 3
                combined += p_b
                
                if not mgr.check_overlap(combined, cutoff=1.2, verbose=False):
                    comp_a = "".join(frag_a.symbols)
                    comp_b = "".join(p_b.symbols)
                    if comp_b == "HH": comp_b = "H2" # Make it more readable
                    
                    combined.info['mechanism'] = f"Generic Single-Site: {comp_a} on {s['index']}, byproduct={comp_b}, rot={angle:.1f}"
                    combined.info['reaction_type'] = 'h_exchange'
                    candidates.append(combined)
                    break
                else:
                    stats['overlap'] += 1
                    
    return candidates


def _execute_generic_dissociation(mgr, molecule, c_idx, ligands, pairs, rot_steps):
    """ Internal subroutine to execute Generic Dissociative Chemisorption on pairs of dangling bonds """
    candidates = []
    stats = {'overlap': 0, 'deduplicated': 0}
    seen_formulas = set()
    
    for l_info in ligands:
        formula = l_info.get('formula', 'Unknown')
        if formula in seen_formulas: 
            stats['deduplicated'] += 1
            continue
        seen_formulas.add(formula)
        
        indices_b = l_info['indices']
        frag_b = molecule[indices_b]
        binding_idx_b = indices_b.index(l_info['binding_atoms'][0])
        
        indices_a = list(set(range(len(molecule))) - set(indices_b))
        frag_a = molecule[indices_a]
        binding_idx_a = indices_a.index(c_idx)
        
        for (s1, s2) in pairs:
            for active_1, active_2 in [(s1, s2), (s2, s1)]:
                best_pose = None
                for angle in np.linspace(0, 360, rot_steps, endpoint=False):
                    p_a = mgr._place_at_dangling_bond(frag_a, binding_idx_a, l_info['bond_vec'], 
                                                       active_1['pos'], active_1['db_vector'], 2.35, rot_angle=angle)
                    
                    bond_len_b = 1.48 if frag_b.symbols[binding_idx_b] == 'H' else 2.1
                    p_b = mgr._place_at_dangling_bond(frag_b, binding_idx_b, -l_info['bond_vec'], 
                                                       active_2['pos'], active_2['db_vector'], bond_len_b, rot_angle=0)
                    
                    combined = mgr.slab.copy()
                    for a in p_a: a.tag = 2
                    combined += p_a
                    for a in p_b: a.tag = 3
                    combined += p_b
                    
                    if not mgr.check_overlap(combined, cutoff=1.2, verbose=False):
                        comp_a = "".join(frag_a.symbols)
                        combined.info['mechanism'] = f"Generic Chemisorption: {comp_a} on {active_1['index']}, {frag_b.symbols[binding_idx_b]} on {active_2['index']}, rot={angle:.1f}"
                        combined.info['reaction_type'] = 'chemisorption'
                        best_pose = combined
                        break
                
                if best_pose:
                    candidates.append(best_pose)
                    break
                else:
                    stats['overlap'] += 1
                    
    return candidates
