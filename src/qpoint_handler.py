import yaml
import numpy as np
import os

class QPointParser:
    """
    Parses Phonopy qpoints.yaml to extract vibrational frequencies and eigenvectors.
    Focused on identifies unstable (imaginary) modes for structural refinement.
    """
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"  [QPointParser] Could not find: {file_path}")
        self.file_path = file_path
        self.data = self._load_yaml()

    def _load_yaml(self):
        """Loads large YAML files safely."""
        with open(self.file_path, 'r') as f:
            # Use CLoader if available for performance
            try:
                from yaml import CLoader as Loader
            except ImportError:
                from yaml import Loader
            return yaml.load(f, Loader=Loader)

    def get_filtered_modes(self, freq_threshold=-0.1, max_modes=None):
        """
        Extracts, filters, and sorts modes based on frequency threshold.
        
        Args:
            freq_threshold (float): Only modes with frequency < threshold are returned.
            max_modes (int): Limit the number of returned modes (most unstable first).
            
        Returns:
            list: List of dicts containing 'frequency' and 'eigenvector' (N_atoms, 3).
        """
        if 'phonon' not in self.data:
            return []

        # We primarily deal with molecules at q=0 (Gamma point)
        # qpoints.yaml structure: phonon -> list of q-points
        qpoint = self.data['phonon'][0]
        bands = qpoint.get('band', [])
        
        modes = []
        for b in bands:
            freq = b.get('frequency')
            if freq is None: continue
            
            if freq < freq_threshold:
                # Parse eigenvector: (N_atoms, [[re, im], [re, im], [re, im]])
                raw_eig = b.get('eigenvector')
                if not raw_eig: continue
                
                num_atoms = len(raw_eig)
                eig_vec = np.zeros((num_atoms, 3))
                for i in range(num_atoms):
                    # Each atom has 3 Cartesian components
                    for j in range(3):
                        # raw_eig[i][j] is [real, imag]
                        eig_vec[i, j] = raw_eig[i][j][0]
                
                modes.append({
                    'frequency': freq,
                    'eigenvector': eig_vec
                })
        
        # Sort by frequency (ascending: most negative/unstable first)
        modes.sort(key=lambda x: x['frequency'])
        
        if max_modes and max_modes > 0:
            modes = modes[:max_modes]
            
        return modes
