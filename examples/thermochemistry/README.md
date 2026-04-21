# Thermochemistry Analysis Example

This example demonstrates how to calculate the **Gibbs Free Energy ($G$)** for molecular and surface species using the frequencies obtained from Phonopy or the `VibrationalAnalyzer`.

## 1. Theoretical Background

The Gibbs free energy is calculated as:
$$G(T) = E_{elec} + ZPE + U_{vib}(T) + H_{corr}(T) - T \cdot S_{total}(T)$$

Where:
- $E_{elec}$: Electronic potential energy (from MLIP/DFT).
- $ZPE$: Zero-Point Energy ($\sum \frac{1}{2} h \nu$).
- $U_{vib}(T)$: Thermal vibrational internal energy.
- $H_{corr}(T)$: Enthalpy correction ($PV$ work and trans/rot internal energy, relevant for gas phase).
- $S_{total}(T)$: Total entropy (Vibrational + Translational + Rotational).

### Modes of Calculation
1.  **Gas Phase (`gas`)**: Includes all degrees of freedom (3D translation, rotation, and vibration). Uses the Sackur-Tetrode equation for $S_{trans}$.
2.  **Adsorbed State (`adsorbent`)**: Assumes all degrees of freedom are vibrational (Harmonic Oscillator). $G \approx E_{elec} + F_{vib}$.
3.  **Substrate (`substrate`)**: Treated similarly to the adsorbed state, but typically used for bulk or slab reference energies.

## 2. Usage

### Prerequisites
Ensure you have the `qpoints.yaml` file generated from a previous vibrational analysis (e.g., from the `mode_following_relaxation` example).

### Basic Command (Adsorbate)
```bash
python run_thermo_analysis.py qpoints.yaml --energy -130.458 --mode adsorbent
```

### Gas Phase Command
For gas phase, you must provide the molecular mass and moments of inertia.
```bash
python run_thermo_analysis.py qpoints.yaml --energy -125.2 --mode gas --mass 133.2 --sigma 1 --moments 450.2 450.2 800.5
```

## 3. Tool Description
- `run_thermo_analysis.py`: The entry point script for users.
- `thermo_engine.py` (in `src/`): The core physics library implementing statistical mechanics formulas.

## 4. References
- McQuarrie, D. A. *Statistical Mechanics*.
- Cramer, C. J. *Essentials of Computational Chemistry*.
