import yaml
import numpy as np
from thermo_engine import ThermoCalculator, GasThermo

class PhonopyParser:
    def __init__(self, yaml_file):
        self.filename = yaml_file
        with open(yaml_file, 'r') as f:
            self.data = yaml.safe_load(f)

    def get_frequencies(self):
        """Extract frequencies in THz from phonopy.yaml"""
        freqs = []
        if 'phonon' in self.data:
            for q_point in self.data['phonon']:
                for band in q_point['band']:
                    freqs.append(band['frequency'])
        return freqs

    def get_molecular_info(self):
        """Try to extract mass and moments of inertia if possible"""
        # Note: Phonopy yaml might not have inertia directly, 
        # but it has structure and mass.
        # This is a placeholder for custom logic if needed.
        pass

class AnalyzeThermo:
    def __init__(self, phonopy_yaml, e_elec_ev=0.0):
        self.parser = PhonopyParser(phonopy_yaml)
        self.freqs = self.parser.get_frequencies()
        self.e_elec_ev = e_elec_ev
        self.e_elec_j_mol = e_elec_ev * 96485.3 # eV to J/mol
        
    def run_analysis(self, T_range, mode='adsorbent', mass_amu=None, sigma=1, moments=None):
        """
        mode: 'adsorbent', 'substrate', or 'gas'
        """
        results = []
        calc = ThermoCalculator(self.freqs)
        
        for T in T_range:
            # Vibrational part (Harmonic Oscillator)
            s_vib = calc.calculate_vib_entropy(T)
            # f_vib = ZPE + U_vib - T*S_vib
            f_vib = calc.calculate_vib_free_energy(T)
            
            if mode == 'gas':
                if mass_amu is None:
                    raise ValueError("mass_amu is required for gas mode")
                
                # Symmetry detection for H-U correction
                symm = 'nonlinear' if (moments and len(moments) == 3) else 'linear'
                h_trans_rot = GasThermo.calculate_enthalpy_correction(T, symmetry=symm)
                
                s_trans = GasThermo.calculate_trans_entropy(mass_amu, T)
                s_rot = GasThermo.calculate_rot_entropy(moments, T, sigma, symmetry=symm) if moments else 0.0
                
                s_total = s_vib + s_trans + s_rot
                # G(gas) = E_elec + ZPE + U_vib + (H_trans+H_rot) - T*(S_vib + S_trans + S_rot)
                # Since f_vib = ZPE + U_vib - T*S_vib, we have:
                g_total = self.e_elec_j_mol + f_vib + h_trans_rot - T * (s_trans + s_rot)
            else:
                # For adsorbates/substrates, G is approximately equal to Helmholtz free energy F
                # G = E_elec + ZPE + U_vib - T*S_vib
                s_total = s_vib
                g_total = self.e_elec_j_mol + f_vib
            
            results.append({
                'T': T,
                'S': s_total,
                'G_kJ_mol': g_total / 1000.0,
                'G_eV': g_total / 96485.3
            })
            
        return results

def print_results(results):
    print(f"{'T (K)':>8} | {'S (J/mol·K)':>12} | {'G (kJ/mol)':>12} | {'G (eV)':>12}")
    print("-" * 55)
    for r in results:
        print(f"{r['T']:8.2f} | {r['S']:12.4f} | {r['G_kJ_mol']:12.4f} | {r['G_eV']:12.6f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate Gibbs Free Energy from Phonopy YAML")
    parser.add_argument("yaml", help="phonopy.yaml or mesh.yaml file")
    parser.add_argument("--energy", type=float, default=0.0, help="Electronic energy (eV)")
    parser.add_argument("--mode", choices=['gas', 'adsorbent', 'substrate'], default='adsorbent', 
                        help="Calculation mode (gas includes trans/rot/PV)")
    parser.add_argument("--mass", type=float, help="Molecular mass (amu) for gas mode")
    parser.add_argument("--sigma", type=int, default=1, help="Symmetry number for gas mode")
    parser.add_argument("--moments", type=float, nargs='+', help="Moments of inertia for gas mode")
    
    args = parser.parse_args()
    
    analyzer = AnalyzeThermo(args.yaml, args.energy)
    temps = [298.15, 400.0, 500.0, 600.0, 700.0, 800.0]
    
    print(f"Analyzing {args.yaml} in '{args.mode}' mode...")
    try:
        res = analyzer.run_analysis(temps, mode=args.mode, mass_amu=args.mass, 
                                     sigma=args.sigma, moments=args.moments)
        print_results(res)
    except Exception as e:
        print(f"Error: {e}")
