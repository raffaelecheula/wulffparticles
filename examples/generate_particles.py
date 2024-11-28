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

# Parameters.
nsamples = 1000
natoms_low = 12000
natoms_high = 16000
esurf_low = 1.00
esurf_high = 1.20
random_multiplier = 1.0
random_seed = 16

# Atoms bulk.
atoms_bulk = bulk('Ni', cubic=True)

# Ase Database.
db_name = 'Ni_new.db'
db_ase = connect(db_name, append=True)

# Reference surface energies.
esurf_ref = {
    (1, 0, 0): 1.11,
    (1, 1, 0): 1.12,
    (1, 1, 1): 1.00,
    (2, 1, 1): 1.14,
}

# Center shifts.
a = atoms_bulk.cell[0,0]
center_shifts = [[0, 0, 0], [a/2, 0, 0], [a/4, a/4, 0], [a/4, a/4, a/4]]

# Get formation energy function.
def get_formation_energy(ncoord_dist):
    # Thermodynamics data.
    e_coh_bulk = -4.82 # [eV]
    a_model_relax = 1.10e-4 # [eV]
    b_model_relax = 3.50 # [-]
    # Calculate cohesive energy.
    e_coh = 0.
    for ii in range(13):
        e_coh += ncoord_dist[ii] * e_coh_bulk * np.sqrt(ii)/np.sqrt(12)
        e_coh += ncoord_dist[ii] * a_model_relax*(12-ii)**b_model_relax
    # Calculate formation energy.
    e_form = e_coh-e_coh_bulk*sum(ncoord_dist)
    return e_form

# Generate random numbers.
np.random.seed(seed=random_seed)
esurf_random_dict = {}
for hkl in esurf_ref:
    esurf_random_dict[hkl] = np.random.uniform(
        low=esurf_low, high=esurf_high, size=nsamples,
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
        surface_energies[hkl] = esurf_ref[hkl]+esurf_random_dict[hkl][ii]
    # Get particle shape.
    particle = AsymmetricParticle(
        surface_energies=surface_energies,
        primitive_structure=atoms_bulk,
        natoms=natoms_random[ii],
        random_multiplier=random_multiplier,
    )
    atoms = particle.get_shifted_atoms(center_shift=center_shifts[center_random[ii]])
    facet_types = get_facet_types(atoms=atoms, particle=particle)
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
    ncoord_dist = np.bincount(ncoords)
    ncoord_str = ",".join([str(ii) for ii in ncoord_dist])
    # Add atoms info.
    natoms = len(atoms)
    diameter = 2*(3/4/np.pi*particle.volume)**(1/3)
    atoms.info = {
        "natoms": natoms,
        "ncoords": ncoords,
        "ncoord_dist": ncoord_dist,
        "miller_indices": list(surface_energies.keys()),
        "surface_energies": list(surface_energies.values()),
        "miller_indices_asymm": list(particle.surface_energies_asymm.keys()),
        "surface_energies_asymm": list(particle.surface_energies_asymm.values()),
        "volume": particle.volume,
        "diameter": diameter,
        "facet_fractions": list(particle.facet_fractions.values()),
    }
    # Write to database.
    if db_ase.count(natoms=natoms, ncoord_str=ncoord_str) == 0:
        # Calculate formation energy.
        energy = get_formation_energy(ncoord_dist=ncoord_dist)
        atoms.calc = SinglePointCalculator(atoms=atoms, energy=energy)
        # Write atoms to database.
        db_ase.write(atoms=atoms, ncoord_str=ncoord_str, data=atoms.info)
        # Print to screen.
        ncoord_print = ",".join([f"{ii:6d}" for ii in ncoord_dist])
        print(f"| {natoms:6d} | {ncoord_print} | {energy/natoms:5.3f} |")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
