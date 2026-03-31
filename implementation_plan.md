# Implementation Plan: Generative ML-driven Surface Adsorption Workflow (Final Design)

This final design integrates support for complex **Metal-Ligand precursors**, a protocol for **Model Fine-Tuning**, and a flexible potential interface.

## User Review Required

> [!IMPORTANT]
> **Metal-Ligand Precursor Support**: Current generative models (trained on OC20) excel at small molecules but may struggle with the steric and electronic complexity of large ALD precursors (e.g., TEMAH, TMA).
> **Transfer Learning Protocol**: I have included a "Model Development" phase. If the baseline AdsorbDiff model fails to propose high-quality sites for a new precursor class, the user can trigger an automated fine-tuning loop using custom DFT/MLFF data.
> **Potential Selection**: The `PotentialFactory` remains central, allowing the use of MACE, CHGNet, or NequIP depending on the system's requirements.

## Workflow Architecture

### 1. Precursor & Slab Handling (RDKit + CatKit)
*   **Metal-Organic Handling**: RDKit will manage the molecular graph and conformers of the metal-ligand system.
*   **Surface Site Analysis**: CatKit provides the symmetry-based baseline (Top, Bridge, Hollow).

### 2. Candidate Generation (Generative + Heuristic)
*   **AdsorbDiff/DiG**: Generates initial poses.
*   **Constraint Checking**: A new module will verify "Metal-Surface" proximity and "Ligand-Surface" steric clashes to filter out unphysical generative proposals.

### 3. Fast Screening (Flexible MLFF)
*   **PotentialFactory**: User selects the backend (default: MACE).
*   **Batch Relaxation**: Candidates are relaxed to the nearest local minima.

### 4. Model Development & Fine-tuning (New Phase)
*   **Trigger**: If $E_{ads}$ diversity is low or structures are unstable.
*   **Step 1**: Run traditional heuristic search (CatKit) + MLFF relaxation for ~200 samples.
*   **Step 2**: Use this "Silver-standard" data to fine-tune the generative model's diffusion coefficients.
*   **Step 3**: Re-run the generative sampler with the updated model.

## Proposed Changes

### [Component] Workflow Framework

#### [NEW] [potential_factory.py](file:///c:/Users/user/Downloads/dev_w_antigravity/auto_surface_reaction/potential_factory.py)
Manages different MLFF/DFT engines.

#### [NEW] [ads_workflow_mgr.py](file:///c:/Users/user/Downloads/dev_w_antigravity/auto_surface_reaction/ads_workflow_mgr.py)
The central orchestrator including the `CandidateGenerator` and `ScreeningEngine`.

#### [NEW] [trainer.py](file:///c:/Users/user/Downloads/dev_w_antigravity/auto_surface_reaction/trainer.py)
A template script for fine-tuning diffusion-based generative models on specific precursor/slab systems.

## Verification Plan

### Manual Verification
1.  Verify the `PotentialFactory` correctly switches between MACE and CHGNet.
2.  Test the `CandidateGenerator` with a dummy metal-ligand structure ($Al(Me)_3$).
3.  Confirm that the ASE Database logs $E_{ads}$ and structural fingerprints correctly.
