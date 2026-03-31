from ase.calculators.emt import EMT
from ase.filters import ExpCellFilter

class PotentialFactory:
    """
    A factory class to initialize and manage Machine Learning Force Fields (MLFFs).
    (Torch-free Demo Version)
    """
    @staticmethod
    def get_calculator(name="emt", device="cpu", **kwargs):
        """Returns an ASE calculator. MACE/CHGNet disabled for this demo."""
        print(f"Initializing {name} calculator (Demo Mode)...")
        # In this demo, we use EMT as a robust fallback
        return EMT()

def get_device():
    return "cpu"
