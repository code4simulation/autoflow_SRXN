import numpy as np
from ase import Atoms
from ase.build import bulk, surface, add_adsorbate
from ase.visualize import view
from ase.io import write

def create_passivated_si100():
    # 1. Create Si bulk and (100) surface
    # Lattice constant for Si is ~5.43 Angstrom
    a = 5.431
    si_bulk = bulk('Si', 'diamond', a=a)
    
    # Create (100) surface with 2x2 supercell and 6 layers
    # layers=6 means 6 atomic layers. For Si(100), alternating layers have different orientations.
    # We want a decent thickness.
    layers = 8
    slab = surface(si_bulk, (1, 0, 0), layers=layers, vacuum=10.0)
    slab = slab.repeat((2, 2, 1))
    
    # 2. Identify top and bottom Si atoms
    z_coords = slab.positions[:, 2]
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    
    bottom_si_indices = np.where(np.abs(z_coords - z_min) < 0.1)[0]
    top_si_indices = np.where(np.abs(z_coords - z_max) < 0.1)[0]
    
    # 3. Passivate Bottom with H (2 H per Si)
    # Bond length Si-H ~ 1.48 A
    # The dangling bonds on Si(100) point in [1, 1, -1] and [-1, -1, -1] directions (relative to surface normal)
    # or similar depending on the layer.
    d_si_h = 1.48
    
    # Layer 0 (bottom) dangling bond directions (approximate tetrahedral)
    # For Si(100), the neighbors are at (+/- a/4, +/- a/4, +/- a/4)
    # If the bottom layer lacks neighbors at -z, we add them.
    vec1 = np.array([a/4, a/4, -a/4])
    vec2 = np.array([-a/4, -a/4, -a/4])
    # Normalize and scale
    vec1 = vec1 / np.linalg.norm(vec1) * d_si_h
    vec2 = vec2 / np.linalg.norm(vec2) * d_si_h
    
    h_atoms = []
    for idx in bottom_si_indices:
        pos = slab.positions[idx]
        h_atoms.append(Atoms('H', positions=[pos + vec1]))
        h_atoms.append(Atoms('H', positions=[pos + vec2]))
        
    for h in h_atoms:
        slab += h
        
    # 4. Passivate Top with O
    # User asked for O passivation. Usually this means Si=O or Si-O-Si.
    # Let's use 1 O atom per Si pointing up (terminal) or 2 O per Si.
    # To keep it simple and symmetric to H (but with O), we can add terminal O.
    # Bond length Si=O ~ 1.5 - 1.6 A
    d_si_o = 1.60
    
    # Top layer dangling bond directions:
    # If the layers alternate, we need to check the layer index or just use the mirror of bottom.
    # For a 8 layer slab, layer 7 (top) should have symmetric dangling bonds to layer 0 but pointing up.
    # However, Si(100) layers alternate [1,1,0] and [1,-1,0] bond directions.
    # For layers=8, the top layer (index 7) orientation:
    # Layer 0, 2, 4, 6 have same orientation.
    # Layer 1, 3, 5, 7 have same orientation.
    # So if layer 0 is [1,1,-1], layer 7 might be [1,-1,1] or similar.
    
    # Let's use a simpler approach: add O at bridge or terminal.
    # User said "O로 passivation". Let's add 1 O atom per dangling bond if possible, 
    # or 1 double-bonded O. Let's do 1 O at a reasonable height.
    
    tvec1 = np.array([a/4, -a/4, a/4])
    tvec2 = np.array([-a/4, a/4, a/4])
    tvec1 = tvec1 / np.linalg.norm(tvec1) * d_si_o
    tvec2 = tvec2 / np.linalg.norm(tvec2) * d_si_o

    o_atoms = []
    for idx in top_si_indices:
        pos = slab.positions[idx]
        # Option A: 2 O atoms per Si (like hydroxyls but just O)
        # Option B: 1 O atom per Si (bridging or terminal)
        # Let's go with 2 O per Si to match the H-passivation pattern.
        o_atoms.append(Atoms('O', positions=[pos + tvec1]))
        o_atoms.append(Atoms('O', positions=[pos + tvec2]))

    for o in o_atoms:
        slab += o
        
    # 5. Save to extxyz
    write('si100_passivated.extxyz', slab, format='extxyz')
    print(f"Structure saved to si100_passivated.extxyz with {len(slab)} atoms.")

if __name__ == "__main__":
    create_passivated_si100()
