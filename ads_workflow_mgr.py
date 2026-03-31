import os
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS
from ase.db import connect
from rdkit import Chem
from rdkit.Chem import AllChem
from potential_factory import PotentialFactory, get_device

class AdsorptionWorkflowManager:
    """
    Main orchestrator for automated surface adsorption screening.
    (Torch-free Demo Version)
    """
    def __init__(self, slab_path, precursor_smiles, db_name="results.db", calculator_name="emt"):
        self.slab = read(slab_path)
        self.precur_smiles = precursor_smiles
        # Delete existing DB to start fresh for demo
        if os.path.exists(db_name):
            os.remove(db_name)
        self.db = connect(db_name)
        self.calc_name = calculator_name
        self.device = get_device()
        
    def generate_rdkit_conformers(self, n_confs=3):
        """Generate diverse molecular conformers of the precursor."""
        print(f"Generating {n_confs} RDKit conformers for SMILES: {self.precur_smiles}...")
        mol = Chem.MolFromSmiles(self.precur_smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=AllChem.ETKDG())
        AllChem.MMFFOptimizeMoleculeConfs(mol)
        return mol

    def generate_heuristic_candidates(self, mol, height=2.5, n_rotations=2):
        """Generate candidates based on heuristic site selection."""
        candidates = []
        Lx, Ly, Lz = self.slab.get_cell_lengths_and_angles()[:3]
        
        # Simplified sites for Si(100) 2x2 supercell (Top and Bridge)
        sites = [(0.25, 0.25), (0.75, 0.75), (0.5, 0.5)] 
        
        from ase import Atoms
        for cid in range(mol.GetNumConformers()):
            conf = mol.GetConformer(cid)
            positions = conf.GetPositions()
            symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            precursor_atoms = Atoms(symbols=symbols, positions=positions)
            
            for sx, sy in sites:
                for rot_idx in range(n_rotations):
                    angle = (360 / n_rotations) * rot_idx
                    trial = self.slab.copy()
                    mol_copy = precursor_atoms.copy()
                    mol_copy.rotate(angle, 'z')
                    
                    com = mol_copy.get_center_of_mass()
                    target_pos = np.array([sx * Lx, sy * Ly, np.max(self.slab.positions[:, 2]) + height])
                    mol_copy.translate(target_pos - com)
                    
                    trial += mol_copy
                    candidates.append(trial)
                    
        return candidates

    def run_screening(self, candidates, fmax=1.0, steps=10):
        """Relax candidates using EMT and log results."""
        calculator = PotentialFactory.get_calculator(self.calc_name, device=self.device)
        print(f"Starting screening of {len(candidates)} candidates...")
        
        for i, atoms in enumerate(candidates):
            print(f"--- Screening Candidate {i+1}/{len(candidates)} ---")
            atoms.calc = calculator
            from ase.constraints import FixAtoms
            z_min = np.min(self.slab.positions[:, 2])
            mask = [atom.index < len(self.slab) and atom.position[2] < z_min + 5 for atom in atoms]
            atoms.set_constraint(FixAtoms(mask=mask))
            
            opt = BFGS(atoms, logfile=None)
            try:
                opt.run(fmax=fmax, steps=steps)
                final_energy = atoms.get_potential_energy()
                self.db.write(atoms, candidate_id=i, energy=final_energy, calc_type=self.calc_name)
                print(f"Candidate {i+1} energy: {final_energy:.4f} eV (Logged to DB)")
            except Exception as e:
                print(f"Optimization failed for candidate {i+1}: {e}")

if __name__ == "__main__":
    slab_path = "si100_reconstructed_passivated.extxyz"
    precur_smiles = "CC(C)N([SiH3])C(C)C" 
    
    mgr = AdsorptionWorkflowManager(slab_path, precur_smiles)
    mol = mgr.generate_rdkit_conformers(n_confs=2)
    candidates = mgr.generate_heuristic_candidates(mol, n_rotations=2)
    print(f"Total candidates generated: {len(candidates)}")
    
    mgr.run_screening(candidates, steps=10)
    print("\nWorkflow Run Successfully!")
