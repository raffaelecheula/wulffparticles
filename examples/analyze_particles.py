# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect
from wulffparticles.analyses import (
    get_probabilities,
    get_sites_concentrations,
    replace_facets,
    remove_self_interfaces,
    calculate_natoms_parameter,
)
from wulffparticles.visualize import (
    plot_particle_size_distribution,
    plot_sites_concentrations,
    plot_formation_energies,
    write_particle_data_to_csv,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

# Parameters.
natoms_low = 5000
natoms_high = 28000
temperature = 800 # [K]
diameter_mean = 67 # [Å]
diameter_stddev = 3 # [Å]

# Plot parameters.
plot_kde = True
plot_conc = True
plot_eform = True
plot_espec = True
write_to_csv = True

# Ase Database.
db_name = 'Ni.db'
db_ase = connect(db_name)
selection = f"{natoms_low}<=natoms<={natoms_high}"

# Get list of ase Atoms from ase database.
atoms_list = []
for id in [aa.id for aa in db_ase.select(selection=selection)]:
    atoms_row = db_ase.get(id=id)
    atoms = atoms_row.toatoms()
    atoms.info = atoms_row.data
    atoms.info.update(atoms_row.key_value_pairs)
    atoms.info["id"] = id
    atoms_list.append(atoms)
print(f"Number of particles: {len(atoms_list)}")

# Get list of energies, number of atoms and diameters.
energy_list = [atoms.get_potential_energy() for atoms in atoms_list]
natoms_list = [atoms.info["natoms"] for atoms in atoms_list]
diameter_list = [atoms.info["diameter"] for atoms in atoms_list]

# Function to calculate the formation energy.
def get_eform_fun(atoms, natoms, deltamu):
    return atoms.get_potential_energy()*natoms/len(atoms)-deltamu*len(atoms)/natoms

# Calculate formation energies.
natoms = 49 # [-]
deltamu = -0.008 # [eV]
eform_list = [get_eform_fun(atoms, natoms, deltamu) for atoms in atoms_list]

# Calculate the Boltzmann probabilities.
probabilities = get_probabilities(eform_list=eform_list, temperature=temperature)

# Calaculate natoms distribution and sites distributions.
sites_conc = get_sites_concentrations(
    atoms_list=atoms_list,
    probabilities=probabilities,
)

# Replace facets and remove self interfaces.
replacements = {
    "210": "110",
    "221": "211",
    "310": "211",
    "311": "211",
    "321": "211",
    "331": "211",
    "421": "211",
    "431": "211",
    "511": "211",
    "531": "211",
    "541": "211",
    "551": "211",
    "731": "211",
    "751": "211",
    "971": "211",
}
for key in sites_conc:
    replace_facets(dictionary=sites_conc[key], replacements=replacements)
    remove_self_interfaces(dictionary=sites_conc[key])

# Plot the particle size distribution (weighted KDE).
if plot_kde is True:
    plot_particle_size_distribution(
        diameter_mean=diameter_mean,
        diameter_stddev=diameter_stddev,
        diameter_list=diameter_list,
        probabilities=probabilities,
    )

# Plot the sites concentrations.
if plot_conc is True:
    plot_sites_concentrations(
        sites_conc=sites_conc,
        plot_fractions=True,
    )

# Plot the formation energies.
if plot_eform is True:
    plot_formation_energies(
        eform_list=eform_list,
        natoms_list=natoms_list,
        diameter_list=diameter_list,
        natoms_low=natoms_low,
        natoms_high=natoms_high,
        specific_eform=False,
        relative_eform=True,
    )

# Plot the formation energies.
if plot_espec is True:
    plot_formation_energies(
        eform_list=energy_list,
        natoms_list=natoms_list,
        diameter_list=diameter_list,
        natoms_low=natoms_low,
        natoms_high=natoms_high,
        specific_eform=True,
        relative_eform=False,
    )

# Write particle data to csv.
if write_to_csv is True:
    write_particle_data_to_csv(
        atoms_list=atoms_list,
        probabilities=probabilities,
        replacements=replacements,
        sites_quantity="fraction",
        filename="particles_data.csv",
    )

# Show the plots.
plt.show()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
