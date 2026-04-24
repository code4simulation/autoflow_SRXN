import unittest
import os
import sys
import numpy as np
from ase.build import molecule
from ase.calculators.emt import EMT

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from potentials import SimulationEngine

class TestPotentials(unittest.TestCase):
    def test_engine_init_emt(self):
        """Test SimulationEngine with EMT backend."""
        config = {
            'engine': {
                'potential': {
                    'backend': 'emt',
                    'device': 'cpu'
                }
            }
        }
        engine = SimulationEngine(config)
        calc = engine.get_calculator()
        self.assertIsInstance(calc, EMT)

    def test_engine_init_mace_fallback(self):
        """Test if SimulationEngine handles missing MACE gracefully."""
        config = {
            'engine': {
                'potential': {
                    'backend': 'mace',
                    'model': 'invalid_path_to_force_failure',
                    'device': 'cpu'
                }
            }
        }
        engine = SimulationEngine(config)
        self.assertEqual(engine.backend, 'mace')

    def test_engine_relaxation(self):
        """Test basic relaxation wrapper using EMT."""
        config = {
            'engine': {
                'potential': {'backend': 'emt'},
                'relaxation': {
                    'fmax': 0.05,
                    'steps': 10,
                    'optimizer': 'BFGS'
                }
            }
        }
        engine = SimulationEngine(config)
        atoms = molecule('H2')
        atoms.calc = engine.get_calculator()
        
        # Move one atom slightly to force relaxation
        atoms.positions[1, 2] += 0.1
        
        initial_fmax = (atoms.get_forces()**2).sum(axis=1).max()**0.5
        engine.relax(atoms)
        final_fmax = (atoms.get_forces()**2).sum(axis=1).max()**0.5
        
        self.assertLess(final_fmax, initial_fmax)

    def test_engine_md(self):
        """Test NVT MD wrapper using EMT."""
        config = {
            'engine': {
                'potential': {'backend': 'emt'},
                'md': {
                    'temperature_K': 300,
                    'timestep_fs': 1.0,
                    'md_steps': 5
                }
            }
        }
        engine = SimulationEngine(config)
        atoms = molecule('H2')
        atoms.calc = engine.get_calculator()
        
        initial_pos = atoms.positions.copy()
        engine.run_md(atoms)
        final_pos = atoms.positions
        
        # Atoms should have moved
        self.assertFalse(np.allclose(initial_pos, final_pos))

if __name__ == '__main__':
    unittest.main()
