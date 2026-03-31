import numpy as np
from ase import Atoms
from ase.build import bulk, surface, add_adsorbate
from ase.io import write
from ase.visualize import view

def create_reconstructed_si100():
    # 1. Setup Base Si(100) Slab
    a = 5.431
    si_bulk = bulk('Si', 'diamond', a=a)
    
    # surface() with (1,0,0) and size=(2,2,6)
    # The default surface unit cell for Si(100) in ASE is primitive (1x1).
    # Its area is (a/sqrt(2) x a/sqrt(2)).
    slab = surface(si_bulk, (1, 0, 0), layers=8, vacuum=15.0)
    slab = slab.repeat((2, 2, 1))
    
    # 2. Identify Top and Bottom layers
    z_coords = slab.positions[:, 2]
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    
    bottom_si_indices = np.where(np.abs(z_coords - z_min) < 0.1)[0]
    top_si_indices = np.where(np.abs(z_coords - z_max) < 0.1)[0]
    
    print(f"Number of surface Si atoms (Top/Bottom): {len(top_si_indices)} / {len(bottom_si_indices)}")
    
    # 3. 2x1 Reconstruction (Top Surface dimerization)
    # In a 2x2 supercell, we have 4 Si atoms on the top layer.
    # We pair them into 2 dimers.
    # The top layer atoms in a 2x2 Si(100) repeat are usually arranged in a grid.
    # We need to find pairs that are neighbors along one direction.
    
    # Sort top atoms by X, then Y
    top_pos = slab.positions[top_si_indices]
    # We'll dimerize along the X direction (or whatever the neighbor direction is)
    # Let's group by Y and then pair in X.
    y_vals = np.unique(np.round(top_pos[:, 1], 3))
    
    dimer_bond_length = 2.35
    dimer_si_pairs = []
    
    for y in y_vals:
        indices_in_row = top_si_indices[np.abs(top_pos[:, 1] - y) < 0.1]
        # Sort these indices by X
        row_x = slab.positions[indices_in_row, 0]
        sorted_indices = indices_in_row[np.argsort(row_x)]
        
        # Pair them: (0,1), (2,3) if exist
        for i in range(0, len(sorted_indices), 2):
            idx1, idx2 = sorted_indices[i], sorted_indices[i+1]
            p1 = slab.positions[idx1]
            p2 = slab.positions[idx2]
            
            # Current distance
            dist = np.linalg.norm(p1 - p2)
            # Shift towards the center to achieve dimer_bond_length
            center = (p1 + p2) / 2
            vec = (p1 - p2) / np.linalg.norm(p1 - p2)
            
            slab.positions[idx1] = center + vec * (dimer_bond_length / 2)
            slab.positions[idx2] = center - vec * (dimer_bond_length / 2)
            dimer_si_pairs.append((idx1, idx2))
            
    print(f"Formed {len(dimer_si_pairs)} dimers on the top surface.")

    # 4. H Passivation (Bottom Surface - Tetrahedral)
    # For Si(100) bottom, each Si has 2 dangling bonds pointing "down" at tetrahedral angles.
    # The vectors to neighbors in a bulk diamond are (+/- a/4, +/- a/4, -a/4).
    d_si_h = 1.48
    
    h_atoms = []
    # Tetrahedral vectors (u,v,w) scaled to bond length
    # Note: diamond neighbors alternate directions by layer.
    # Since we are at the bottom layer (layer 0), we use the directions that Si would have bonds to neighbors below.
    # For Si(100), if the top surface has dimers in X, the neighbors below might be in +/-X, +/-Y.
    # Actually, let's use the exact symmetry.
    
    vec1 = np.array([a/4, a/4, -a/4])
    vec2 = np.array([-a/4, -a/4, -a/4])
    # Or [a/4, -a/4, -a/4] and [-a/4, a/4, -a/4] depending on layer.
    # Let's try to detect the "existing" neighbors to decide the direction.
    
    for idx in bottom_si_indices:
        pos = slab.positions[idx]
        # Check existing Si neighbors (at layer 1)
        # Neighbors at +z
        # We want to place H at -z
        
        # A simple robust way: H1 = (a/4, a/4, -a/4), H2 = (-a/4, -a/4, -a/4)
        # rotated 90 deg if the second layer has neighbors in a different orientation.
        # For Si(100) first layer, neighbors are at [a/4, a/4, a/4] etc.
        # So H should be at [a/4, -a/4, -a/4] and [-a/4, a/4, -a/4].
        
        # Let's verify tetrahedrality: 109.5 deg.
        h_vec1 = np.array([a/4, -a/4, -a/4])
        h_vec2 = np.array([-a/4, a/4, -a/4])
        h_vec1 = h_vec1 / np.linalg.norm(h_vec1) * d_si_h
        h_vec2 = h_vec2 / np.linalg.norm(h_vec2) * d_si_h
        
        slab += Atoms('H', positions=[pos + h_vec1])
        slab += Atoms('H', positions=[pos + h_vec2])
        
    # 5. O Bridge (Top Surface)
    # Place O in the bridge of each dimer.
    d_si_o = 1.65 # Si-O bond length
    for idx1, idx2 in dimer_si_pairs:
        p1 = slab.positions[idx1]
        p2 = slab.positions[idx2]
        center = (p1 + p2) / 2
        # Distance between Si atoms is dimer_bond_length (2.35)
        # Height of O above the Si-Si line:
        # h^2 + (dist/2)^2 = d_si_o^2
        h_sq = d_si_o**2 - (dimer_bond_length / 2)**2
        o_height = np.sqrt(h_sq)
        
        slab += Atoms('O', positions=[center + np.array([0, 0, o_height])])
        
    # 6. Save and Verify
    write('si100_reconstructed_passivated.extxyz', slab, format='extxyz')
    print("Structure saved to si100_reconstructed_passivated.extxyz")
    
    # Simple angle verification for H
    # Find one Si and its two H neighbors
    si_idx = bottom_si_indices[0]
    # H atoms were added last
    h_indices = [len(slab)-8, len(slab)-7] # roughly
    # Better: find atoms with d < 1.6
    from ase.neighborlist import neighbor_list
    i, j = neighbor_list('ij', slab, 1.6)
    my_h = j[i == si_idx]
    if len(my_h) >= 2:
        v1 = slab.positions[my_h[0]] - slab.positions[si_idx]
        v2 = slab.positions[my_h[1]] - slab.positions[si_idx]
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        print(f"Verified H-Si-H angle: {angle:.2f} degrees")

if __name__ == "__main__":
    create_reconstructed_si100()
