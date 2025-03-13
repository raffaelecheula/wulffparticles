# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

from ase.build import bulk
from ase.db import connect
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

# Ase Database.
db_name = 'Ni.db'
db_ase = connect(db_name)
id = 2532

# Get list of ase Atoms from ase database.
atoms_row = db_ase.get(id=id)
atoms = atoms_row.toatoms()
atoms.info = atoms_row.data
atoms.info.update(atoms_row.key_value_pairs)

# Surface energies.
surface_energies = {
    tuple(miller_index): surface_energy for (miller_index, surface_energy) 
    in zip(atoms.info["miller_indices"], atoms.info["surface_energies"])
}
surface_energies_asymm = {
    tuple(miller_index): surface_energy for (miller_index, surface_energy) 
    in zip(atoms.info["miller_indices_asymm"], atoms.info["surface_energies_asymm"])
}

# Get particle.
particle = AsymmetricParticle(
    surface_energies=surface_energies,
    primitive_structure=atoms_bulk,
    natoms=atoms.info["natoms"],
    asymm_multiplier=None,
    surface_energies_asymm=surface_energies_asymm,
)
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
    atoms_scale=0.95,
    atoms_radii=300,
    filename="particle.png",
)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
