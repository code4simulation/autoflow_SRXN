import numpy as np

# Physical constants (CODATA 2018)
kB = 1.380649e-23  # Boltzmann constant [J/K]
h = 6.62607015e-34 # Planck constant [J·s]
R = 8.314462618   # Gas constant [J/(mol·K)]
Na = 6.02214076e23 # Avogadro constant [1/mol]
c = 299792458     # Speed of light [m/s]

def thz_to_cm1(thz):
    """Convert THz to cm^-1"""
    return thz * 1e12 / (c * 100)

def thz_to_joule(thz):
    """Convert THz to Joules"""
    return h * thz * 1e12

class ThermoCalculator:
    def __init__(self, frequencies_thz, temp_range=None):
        """
        frequencies_thz: list or array of vibrational frequencies in THz.
        temp_range: list or array of temperatures in Kelvin.
        """
        self.freqs = np.array([f for f in frequencies_thz if f > 0]) # Ignore zero/imaginary
        self.temps = np.array(temp_range) if temp_range is not None else np.array([298.15])
        
    def calculate_zpe(self):
        """Zero Point Energy in J/mol"""
        zpe = 0.5 * np.sum(h * self.freqs * 1e12) * Na
        return zpe # J/mol

    def calculate_vib_internal_energy(self, T):
        """Thermal vibrational internal energy U_vib(T) in J/mol"""
        if T < 1e-6:
            return 0.0
        
        hv = h * self.freqs * 1e12
        exp_factor = np.exp(hv / (kB * T))
        u_vib = np.sum(hv / (exp_factor - 1)) * Na
        return u_vib

    def calculate_vib_entropy(self, T):
        """Vibrational entropy S_vib(T) in J/(mol·K)"""
        if T < 1e-6:
            return 0.0
        
        hv = h * self.freqs * 1e12
        x = hv / (kB * T)
        # Avoid log(0) for high frequencies
        s_vib = R * np.sum(x / (np.exp(x) - 1) - np.log(1 - np.exp(-x) + 1e-16))
        return s_vib

    def calculate_vib_free_energy(self, T):
        """Vibrational Helmholtz free energy F_vib(T) in J/mol"""
        zpe = self.calculate_zpe()
        u_vib = self.calculate_vib_internal_energy(T)
        s_vib = self.calculate_vib_entropy(T)
        return zpe + u_vib - T * s_vib

class GasThermo:
    @staticmethod
    def calculate_enthalpy_correction(T, symmetry='nonlinear'):
        """
        Calculates the H - U correction for an ideal gas.
        H = U + PV = U + RT
        U = U_trans + U_rot + U_vib + U_elec
        U_trans = 1.5 RT
        U_rot = 1.5 RT (nonlinear) or 1.0 RT (linear)
        H = (1.5 + 1.5 + 1) RT = 4.0 RT (nonlinear)
        H = (1.5 + 1.0 + 1) RT = 3.5 RT (linear)
        This function returns the (H_trans + H_rot) part:
        H_gas = U_vib + U_elec + (H_trans + H_rot)
        """
        if symmetry == 'linear':
            # H_trans + H_rot = 1.5RT + (1.0RT + RT) = 3.5RT
            # Wait, H = U + RT. 
            # U_trans = 1.5RT -> H_trans = 2.5RT
            # U_rot = 1.0RT -> H_rot = 1.0RT (PV term is already in H_trans)
            return 3.5 * R * T
        else:
            # H_trans + H_rot = 2.5RT + 1.5RT = 4.0RT
            return 4.0 * R * T

    @staticmethod
    def calculate_trans_entropy(mass_amu, T, P=101325):
        """
        Sackur-Tetrode equation for translational entropy.
        mass_amu: molecular mass in amu.
        T: Temperature in K.
        P: Pressure in Pa (default 1 atm).
        """
        M = mass_amu * 1.660539e-27 # kg
        # S = R * [ln( (2*pi*M*kB*T/h^2)^1.5 * kB*T/P ) + 2.5]
        prefactor = (2 * np.pi * M * kB * T / (h**2))**1.5
        s_trans = R * (np.log(prefactor * kB * T / P) + 2.5)
        return s_trans

    @staticmethod
    def calculate_rot_entropy(moments_amu_ang2, T, sigma, symmetry='nonlinear'):
        """
        Rotational entropy.
        moments_amu_ang2: list of moments of inertia [I1, I2, I3] in amu*Ang^2.
        T: Temperature in K.
        sigma: Symmetry number.
        """
        # Convert amu*Ang^2 to kg*m^2
        conv = 1.660539e-27 * (1e-10**2)
        I = [m * conv for m in moments_amu_ang2]
        
        if symmetry == 'linear':
            # S = R * [ln(8*pi^2*I*kB*T / (sigma*h^2)) + 1]
            s_rot = R * (np.log(8 * np.pi**2 * I[0] * kB * T / (sigma * h**2)) + 1)
        else:
            # S = R * [ln(sqrt(pi)/sigma * (8*pi^2*kB*T/h^2)^1.5 * sqrt(I1*I2*I3)) + 1.5]
            product_I = np.prod(I)
            term1 = np.sqrt(np.pi) / sigma
            term2 = (8 * np.pi**2 * kB * T / (h**2))**1.5
            s_rot = R * (np.log(term1 * term2 * np.sqrt(product_I)) + 1.5)
            
        return s_rot

if __name__ == "__main__":
    # Quick test
    test_freqs = [10.0, 20.0, 30.0] # THz
    calc = ThermoCalculator(test_freqs)
    T = 298.15
    print(f"ZPE: {calc.calculate_zpe()/1000:.3f} kJ/mol")
    print(f"S_vib: {calc.calculate_vib_entropy(T):.3f} J/(mol·K)")
    print(f"F_vib: {calc.calculate_vib_free_energy(T)/1000:.3f} kJ/mol")
