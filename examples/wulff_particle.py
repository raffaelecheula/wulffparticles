# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.build import bulk
from ase.db import connect
from ase.neighborlist import NeighborList
from ase.neighborlist import natural_cutoffs
from ase.calculators.singlepoint import SinglePointCalculator
from wulffpack import SingleCrystal, Icosahedron, Decahedron

from wulffparticles.asymmetric_particle import AsymmetricParticle
from wulffparticles.facet_types import get_facet_types, write_facet_types

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

# Atoms bulk.
atoms_bulk = bulk('Ni', cubic=True)

# Reference surface energies.
surface_energies = {
    (1, 0, 0): 1.05,
    (1, 1, 0): 1.02,
    (1, 1, 1): 1.00,
    (2, 1, 1): 1.02,
}

# Get particle.
particle = AsymmetricParticle(
    surface_energies=surface_energies,
    primitive_structure=atoms_bulk,
    natoms=20000,
    random_multiplier=1.0,
)
atoms = particle.get_shifted_atoms(center_shift=None)
# Calculate facet types.
facet_types = get_facet_types(
    atoms=atoms,
    particle=particle,
    primitive_structure=atoms_bulk,
)
# Write facet types.
write_facet_types(atoms=atoms, facet_types=facet_types, filename="atoms.png")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
