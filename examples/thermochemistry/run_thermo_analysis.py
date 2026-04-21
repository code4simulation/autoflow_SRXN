import os
import sys
import yaml
import numpy as np
import argparse

# Add the src directory to Python path for local development
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from thermo_engine import ThermoCalculator, GasThermo, eV_to_J_mol
from qpoint_handler import QPointParser

class AnalyzeThermo:
    """
    Main analyst class for processing Phonopy qpoints.yaml into Gibbs Free Energy.
    """
    def __init__(self, qpoints_yaml, e_elec_ev=0.0):
        self.parser = QPointParser(qpoints_yaml)
        self.e_elec_ev = e_elec_ev
        self.e_elec_j_mol = e_elec_ev * eV_to_J_mol
        
        # Load all frequencies (THz)
        self.freqs = []
        for phon in self.parser.data['phonon']:
            for band in phon['band']:
                self.freqs.append(band['frequency'])
        
    def run_analysis(self, T_range, mode='adsorbent', mass_amu=None, sigma=1, moments=None):
        """
        Calculates thermochemical properties for a given temperature range.
        
        Args:
            mode: 'gas', 'adsorbent', or 'substrate'.
            mass_amu: Molecular mass (required for 'gas').
            sigma: Symmetry number (required for 'gas' entropy).
            moments: Moments of inertia (required for 'gas' rot entropy).
        """
        results = []
        calc = ThermoCalculator(self.freqs)
        
        for T in T_range:
            # Vibrational contribution (Helmholtz free energy includes ZPE)
            s_vib = calc.calculate_vib_entropy(T)
            f_vib = calc.calculate_vib_free_energy(T)
            
            if mode == 'gas':
                if mass_amu is None:
                    raise ValueError("mass_amu is required for 'gas' mode.")
                
                # Detect symmetry for linear/nonlinear logic
                symm = 'nonlinear' if (moments and len(moments) == 3) else 'linear'
                
                # Gas phase corrections: H_corr = H_trans + H_rot
                h_corr = GasThermo.calculate_enthalpy_correction(T, symmetry=symm)
                s_trans = GasThermo.calculate_trans_entropy(mass_amu, T)
                s_rot = GasThermo.calculate_rot_entropy(moments, T, sigma, symmetry=symm) if moments else 0.0
                
                s_total = s_vib + s_trans + s_rot
                # G_gas = E_elec + F_vib + H_corr - T*(S_trans + S_rot)
                g_total = self.e_elec_j_mol + f_vib + h_corr - T * (s_trans + s_rot)
            else:
                # For adsorbents and substrates, G is approx Helmholtz F in the solid phase
                s_total = s_vib
                g_total = self.e_elec_j_mol + f_vib
            
            results.append({
                'T': T,
                'S': s_total,
                'G_kJ_mol': g_total / 1000.0,
                'G_eV': g_total / eV_to_J_mol
            })
            
        return results

def print_table(results, title="Thermochemistry"):
    print(f"\n--- {title} ---")
    print(f"{'T (K)':>8} | {'S (J/mol·K)':>12} | {'G (kJ/mol)':>12} | {'G (eV)':>12}")
    print("-" * 55)
    for r in results:
        print(f"{r['T']:8.2f} | {r['S']:12.4f} | {r['G_kJ_mol']:12.4f} | {r['G_eV']:12.6f}")

def main():
    # Use first argument as config path or default to local config.yaml
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    
    if not os.path.exists(input_path):
        print(f"Error: Configuration file '{input_path}' not found.")
        print("Usage: python run_thermo_analysis.py [config.yaml]")
        return

    # Load All Settings from YAML (No argparse)
    with open(input_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if not config or 'thermochemistry' not in config:
        print("Error: Invalid configuration format. Missing 'thermochemistry' section.")
        return
        
    cfg = config['thermochemistry']
    
    # Resolve Path for qpoints
    base_dir = os.path.dirname(os.path.abspath(input_path))
    q_rel = cfg.get('qpoints_file', 'qpoints.yaml')
    qpoints_file = os.path.normpath(os.path.join(base_dir, q_rel))
    
    if not os.path.exists(qpoints_file):
        print(f"Error: Vibrational data '{qpoints_file}' not found.")
        return

    # Extract parameters
    e_elec = cfg.get('electronic_energy', 0.0)
    mode = cfg.get('mode', 'adsorbent')
    temps = cfg.get('temperature_range', [298.15])
    
    gas_cfg = cfg.get('gas_properties', {})
    mass = gas_cfg.get('mass')
    sigma = gas_cfg.get('sigma', 1)
    moments = gas_cfg.get('moments')

    # Execute Analysis
    analyzer = AnalyzeThermo(qpoints_file, e_elec)
    
    print(f"\n{'='*20} AutoFlow-SRXN THERMO ANALYSIS {'='*20}")
    print(f"  Configuration: {input_path}")
    print(f"  Input YAML:    {qpoints_file}")
    print(f"  Mode:          {mode}")
    print(f"  E_elec:        {e_elec:.6f} eV")
    
    if mode == 'gas':
        print(f"  Gas Prop:      mass={mass} amu, sigma={sigma}, moments={moments}")
    
    try:
        results = analyzer.run_analysis(
            T_range=temps,
            mode=mode,
            mass_amu=mass,
            sigma=sigma,
            moments=moments
        )
        print_table(results, title=f"Results ({mode})")
    except Exception as e:
        print(f"Error during calculation: {e}\n")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
