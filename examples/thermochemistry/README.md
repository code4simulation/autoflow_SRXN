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

### Execution
The tool is now entirely configuration-driven. All parameters such as energy, mass, and symmetry must be defined in your `config.yaml`.

```bash
# Execute with a specific config
python run_thermo_analysis.py config_H2.yaml

# Execute with default config.yaml in the current directory
python run_thermo_analysis.py
```

## 3. Configuration Format
Your YAML file must contain a `thermochemistry` section as shown below:

```yaml
thermochemistry:
  qpoints_file: "sample_data/qpoints_H2.yaml"
  electronic_energy: -6.772
  mode: "gas"
  temperature_range: [298.15, 500, 1000]
  gas_properties:
    mass: 2.016
    sigma: 2
    moments: [0.280]
```

## 3. Tool Description
- `run_thermo_analysis.py`: The entry point script for users.
- `thermo_engine.py` (in `src/`): The core physics library implementing statistical mechanics formulas.

## 4. References
- McQuarrie, D. A. *Statistical Mechanics*.
- Cramer, C. J. *Essentials of Computational Chemistry*.
