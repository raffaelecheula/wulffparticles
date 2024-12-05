# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

from typing import Dict, Tuple
import numpy as np
from ase import Atoms
from wulffpack.core import BaseParticle
from wulffpack.core.form import Form, setup_forms
from wulffpack.core.geometry import (
    get_symmetries,
    break_symmetry,
    get_angle,
    get_rotation_matrix,
    get_standardized_structure,
)

# -------------------------------------------------------------------------------------
# ASYMMETRIC PARTICLE
# -------------------------------------------------------------------------------------

class AsymmetricParticle(BaseParticle):

    def __init__(
        self,
        surface_energies: Dict[tuple, float],
        primitive_structure: Atoms = None,
        natoms: int = 1000,
        asymm_multiplier: float = 1.0,
        symprec: float = 1e-5,
        tol: float = 1e-5,
    ):
        standardized_structure = get_standardized_structure(
            structure=primitive_structure,
            symprec=symprec,
        )
        full_symmetries = get_symmetries(
            structure=standardized_structure,
            symprec=symprec,
        )
        symmetries = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])]
        
        surface_energies_asymm = {}
        parent_miller_indices = {}
        for miller_indices in surface_energies.keys():
            for sym in full_symmetries:
                hkl = tuple([int(ii) for ii in np.dot(sym, miller_indices)])
                if hkl not in surface_energies_asymm.keys():
                    surface_energies_asymm[hkl] = surface_energies[miller_indices]
                    parent_miller_indices[hkl] = miller_indices
        
        for miller_indices in surface_energies_asymm.keys():
            surface_energies_asymm[miller_indices] *= np.random.uniform(
                low=1.0, high=asymm_multiplier,
            )
        
        forms = []
        for miller_indices, energy in surface_energies_asymm.items():
            forms.append(Form(
                miller_indices=miller_indices,
                energy=energy,
                cell=standardized_structure.cell.T,
                symmetries=symmetries,
                parent_miller_indices=parent_miller_indices[miller_indices],
            ))

        super().__init__(
            forms=forms,
            standardized_structure=standardized_structure,
            natoms=natoms,
            tol=tol,
        )
        self.surface_energies_asymm = surface_energies_asymm

    @property
    def atoms(self):
        return self._get_atoms()

    @property
    def diameter(self):
        return 2*(3/4/np.pi*self.volume)**(1/3)

    def get_shifted_atoms(
        self,
        center_shift: Tuple[float, float, float] = None,
    ) -> Atoms:
        return self._get_atoms(center_shift=center_shift)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
