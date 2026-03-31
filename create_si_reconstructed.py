import numpy as np
from ase import Atoms
from ase.build import diamond100
from ase.io import write
from surface_utils import passivate_slab, reconstruct_2x1_buckled
import os

def build_flexible_si100_slab(size=(3, 3, 8)):
    """
    Demonstrate robust reconstruction and passivation on a larger or rotated cell.
    """
    print(f"Building Si(100) slab with size={size}...")
    # 1. Base Slab
    slab = diamond100('Si', size=size, vacuum=15.0)
    
    # 2. Apply Reconstruction (BEFORE passivation to set Si-Si pairs)
    dimer_pairs = reconstruct_2x1_buckled(slab, bond_length=2.30, buckle=0.7)
    
    # 3. Apply H-Passivation to Bottom Surface
    # Automatically finds dangling bonds and applies tetrahedral geometry
    slab = passivate_slab(slab, species='H', side='bottom')
    
    # 4. Filter top surface Si for Oxygen placement (those that were dimerized)
    # Actually, we can just use the dimer_pairs returned
    o_offset = 1.5
    for idx1, idx2 in dimer_pairs:
        p1, p2 = slab.positions[idx1], slab.positions[idx2]
        center = (p1 + p2) / 2
        # Place O at the center with a 1.5A offset
        slab += Atoms('O', positions=[center + np.array([0, 0, o_offset])])
        
    # 5. Save final result
    write('si100_reconstructed_passivated.extxyz', slab, format='extxyz')
    print(f"Final structure with {len(slab)} atoms saved to extxyz.")

if __name__ == "__main__":
    # Test on a 3x3 supercell (standard is 2x2)
    build_flexible_si100_slab(size=(3, 3, 8))
