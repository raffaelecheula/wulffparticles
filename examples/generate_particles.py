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
from wulffparticles.sites_names import get_sites_hkl, get_sitenames_distribution_types

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

# Parameters.
nsamples = 1000
natoms_low = 8000
natoms_high = 22000
symm_multiplier = 1.05
asymm_multiplier = 1.05
random_seed = None

# Atoms bulk.
atoms_bulk = bulk('Ni', cubic=True)

# Ase Database.
db_name = 'Ni.db'
db_ase = connect(db_name, append=True)

# Reference surface energies.
esurf_ref = {
    (1, 0, 0): 1.08,
    (1, 1, 0): 1.10 * 2.00,
    (1, 1, 1): 1.00,
    (2, 1, 1): 1.10 * 0.95,
}

# Center shifts.
a = atoms_bulk.cell[0,0]
center_shifts = [[0, 0, 0], [a/2, 0, 0], [a/4, a/4, 0], [a/4, a/4, a/4]]

# Get formation energy function.
def get_formation_energy(ncoord_distrib):
    # Thermodynamics data.
    e_coh_bulk = -4.82 # [eV]
    a_model_relax = 1.10e-4 # [eV]
    b_model_relax = 3.50 # [-]
    # Calculate cohesive energy.
    e_coh = 0.
    for ii in range(13):
        e_coh += ncoord_distrib[ii] * e_coh_bulk * np.sqrt(ii)/np.sqrt(12)
        e_coh += ncoord_distrib[ii] * a_model_relax*(12-ii)**b_model_relax
    # Calculate formation energy.
    e_form = e_coh-e_coh_bulk*sum(ncoord_distrib)
    return e_form

# Generate random numbers.
np.random.seed(seed=random_seed)
symm_multiplier_dict = {}
for hkl in esurf_ref:
    symm_multiplier_dict[hkl] = np.random.uniform(
        low=1, high=symm_multiplier, size=nsamples,
    )
natoms_random = np.random.randint(low=natoms_low, high=natoms_high, size=nsamples)
center_random = np.random.randint(low=0, high=len(center_shifts), size=nsamples)

# Generate particles.
ncoords_str = "".join([f"{ii+1:6d} " for ii in range(12)])
print("-"*111+"\n| natoms | ncoords"+ncoords_str+"| espec |\n"+"-"*111)
for ii in range(nsamples):
    # Get surface energies.
    surface_energies = {}
    for hkl in esurf_ref:
        surface_energies[hkl] = esurf_ref[hkl]*symm_multiplier_dict[hkl][ii]
    # Get particle shape.
    particle = AsymmetricParticle(
        surface_energies=surface_energies,
        primitive_structure=atoms_bulk,
        natoms=natoms_random[ii],
        asymm_multiplier=asymm_multiplier,
    )
    atoms = particle.get_shifted_atoms(center_shift=center_shifts[center_random[ii]])
    sites_hkl = get_sites_hkl(
        atoms=atoms,
        particle=particle,
        primitive_structure=atoms_bulk,
    )
    sites_distrib = get_sitenames_distribution_types(sites_hkl=sites_hkl)
    # Calculate coordination numbers.
    nlist = NeighborList(
        cutoffs=natural_cutoffs(atoms),
        skin=0.3,
        sorted=False,
        self_interaction=False,
        bothways=True,
    )
    nlist.update(atoms)
    ncoords = np.sum(nlist.get_connectivity_matrix(sparse=False), axis=1)
    ncoord_distrib = np.bincount(ncoords)
    ncoord_str = ",".join([str(ii) for ii in ncoord_distrib])
    # Add atoms info.
    natoms = len(atoms)
    atoms.info = {
        "natoms": natoms,
        "ncoords": ncoords,
        "ncoord_distrib": ncoord_distrib,
        "miller_indices": list(surface_energies.keys()),
        "surface_energies": list(surface_energies.values()),
        "miller_indices_asymm": list(particle.surface_energies_asymm.keys()),
        "surface_energies_asymm": list(particle.surface_energies_asymm.values()),
        "volume": particle.volume,
        "diameter": particle.diameter,
        "facet_fractions": list(particle.facet_fractions.values()),
        "sites_distrib_facets": sites_distrib["facets"],
        "sites_distrib_edges": sites_distrib["edges"],
        "sites_distrib_corners": sites_distrib["corners"],
    }
    # Write to database.
    if db_ase.count(natoms=natoms, ncoord_str=ncoord_str) == 0:
        # Calculate formation energy.
        energy = get_formation_energy(ncoord_distrib=ncoord_distrib)
        atoms.calc = SinglePointCalculator(atoms=atoms, energy=energy)
        # Write atoms to database.
        db_ase.write(atoms=atoms, ncoord_str=ncoord_str, data=atoms.info)
        # Print to screen.
        ncoord_print = ",".join([f"{ii:6d}" for ii in ncoord_distrib])
        print(f"| {natoms:6d} | {ncoord_print} | {energy/natoms:5.3f} |")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
