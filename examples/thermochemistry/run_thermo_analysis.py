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
    parser = argparse.ArgumentParser(description="Calculate Gibbs Free Energy from qpoints.yaml")
    parser.add_argument("yaml", help="qpoints.yaml file from AutoFlow-SRXN or Phonopy")
    parser.add_argument("--energy", type=float, default=0.0, help="Electronic energy E_elec (eV)")
    parser.add_argument("--mode", choices=['gas', 'adsorbent', 'substrate'], default='adsorbent', 
                        help="Calculation mode")
    parser.add_argument("--mass", type=float, help="Molecular mass (amu) - required for gas mode")
    parser.add_argument("--sigma", type=int, default=1, help="Symmetry number for gas mode")
    parser.add_argument("--moments", type=float, nargs='+', help="Moments of inertia for gas mode")
    parser.add_argument("--temps", type=float, nargs='+', default=[298.15, 400, 500, 600, 700, 800],
                        help="Temperatures to evaluate (K)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.yaml):
        print(f"Error: File {args.yaml} not found.")
        return

    analyzer = AnalyzeThermo(args.yaml, args.energy)
    
    print(f"Analyzing {args.yaml}...")
    print(f"  Mode: {args.mode}")
    print(f"  E_elec: {args.energy:.6f} eV")
    
    try:
        results = analyzer.run_analysis(
            T_range=args.temps,
            mode=args.mode,
            mass_amu=args.mass,
            sigma=args.sigma,
            moments=args.moments
        )
        print_table(results, title=f"Results ({args.mode})")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
