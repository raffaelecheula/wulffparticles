# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

from ase.build import bulk
from wulffparticles.asymmetric_particle import AsymmetricParticle
from wulffparticles.sites_names import (
    get_sites_hkl,
    get_sitenames_distribution_types,
)
from wulffparticles.visualize import (
    write_atoms_and_sites_hkl,
    write_particle_and_atoms,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

# Atoms bulk.
atoms_bulk = bulk('Ni', cubic=True)

# Reference surface energies.
surface_energies = {
    (1, 0, 0): 1.05,
    (1, 1, 0): 1.08,
    (1, 1, 1): 1.00,
    #(2, 1, 0): 1.10,
    (2, 1, 1): 1.08,
    #(3, 1, 1): 1.10,
    #(3, 2, 1): 1.10,
    #(3, 3, 1): 1.10,
}

# Get particle.
particle = AsymmetricParticle(
    surface_energies=surface_energies,
    primitive_structure=atoms_bulk,
    natoms=2000,
    asymm_multiplier=1.0,
)
atoms = particle.get_shifted_atoms(center_shift=None)
print(f"Number of atoms: {len(atoms)}")
print(f"Diameter: {particle.diameter:.2f} Å")

# Calculate sites hkl.
sites_hkl = get_sites_hkl(
    atoms=atoms,
    particle=particle,
    primitive_structure=atoms_bulk,
)

# Write atoms and sites hkl.
write_atoms_and_sites_hkl(atoms=atoms, sites_hkl=sites_hkl, filename="atoms.png")

# Get sites distribution types.
sits_distrib_dict = get_sitenames_distribution_types(sites_hkl=sites_hkl)

# Write particle and atoms.
write_particle_and_atoms(
    particle=particle,
    atoms=atoms,
    atoms_scale=0.97,
    atoms_radii=400,
    filename="particle.png",
)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
