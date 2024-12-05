# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from scipy.optimize import curve_fit
from ase.db import connect
from wulffparticles.analyses import (
    get_probabilities,
    get_sites_concentrations,
    replace_facets,
    remove_self_interfaces,
    calculate_natoms_parameter,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

# Parameters.
natoms_low = 8000 # [-]
natoms_high = 22000 # [-]
temperature = 800 # [K]
diameter_mean = 67 # [Å]
diameter_stddev = 3 # [Å]

# Plot parameters.
plot_kde = True
plot_conc = True
plot_deltaeform = True
plot_espec = True

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

# Calculate the weighted KDE.
kde = gaussian_kde(dataset=diameter_list, weights=probabilities)

# Calaculate natoms distribution and sites distributions.
sites_conc = get_sites_concentrations(
    atoms_list=atoms_list,
    probabilities=probabilities,
)

# Replace facets and remove self interfaces.
replacements = {
    "210": "110",
    "311": "211",
    "221": "211",
    "331": "211",
    "531": "211",
}
for key in sites_conc:
    replace_facets(dictionary=sites_conc[key], replacements=replacements)
    remove_self_interfaces(dictionary=sites_conc[key])

# Plot the weighted KDE.
if plot_kde is True:
    fig, ax = plt.subplots(figsize=(10, 6))
    x_min = diameter_mean - 4 * diameter_stddev
    x_max = diameter_mean + 4 * diameter_stddev
    x_vect = np.linspace(x_min, x_max, 1000)
    # Plot a reference normal distribution.
    y_norm = norm.pdf(x_vect, diameter_mean, diameter_stddev)
    ax.plot(x_vect, y_norm, color='grey', ls="--")
    # Plot the weighted KDE distribution.
    y_kde = kde(x_vect)
    ax.plot(x_vect, y_kde, color='black')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, np.max(np.concatenate([y_norm, y_kde]))+0.02)
    ax.set_xlabel('Diameter [Å]')
    ax.set_ylabel('Particle density [-]')

# Plot the site concentrations.
if plot_conc is True:
    plot_fractions = True
    x_vect = np.array([ii for ii in sites_conc["facets"]])
    height = np.array([sites_conc["facets"][ii] for ii in sites_conc["facets"]])
    scale_factor = 1/np.sum(height) if plot_fractions else 1
    height *= scale_factor
    sorted_index = np.argsort(-height)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x=x_vect[sorted_index], height=height[sorted_index], color="orangered")
    x_vect = np.array([ii for ii in sites_conc["edges"]])
    height = np.array([sites_conc["edges"][ii] for ii in sites_conc["edges"]])
    height *= scale_factor
    sorted_index = np.argsort(-height)
    ax.bar(x=x_vect[sorted_index], height=height[sorted_index], color="seagreen")
    ax.set_xlabel('Sites types [-]')
    if plot_fractions:
        ax.set_ylabel('Sites fraction [-]')
    else:
        ax.set_ylabel('Sites concentration [N$_{sites}$/N$_{atoms}$]')

# Plot the formation energies.
if plot_deltaeform is True:
    fig, ax = plt.subplots(figsize=(10, 6))
    deltaeform_list = [eform-min(eform_list) for eform in eform_list]
    ax.scatter(natoms_list, deltaeform_list, color='blue', s=30, alpha=0.5)
    ax.set_xlabel('Number of atoms [-]')
    ax.set_ylabel('Relative formation energy [eV]')
    ax.set_xlim(natoms_low, natoms_high)
    # Calculate diameter vs natoms function.
    fun_natoms = lambda diam, a_fit: a_fit*np.array(diam)**3
    a_fit = curve_fit(f=fun_natoms, xdata=diameter_list, ydata=natoms_list)[0][0]
    fun_for = lambda diam: a_fit*np.array(diam)**3
    fun_rev = lambda natoms: np.array(natoms/a_fit)**(1/3)
    ax2 = ax.secondary_xaxis('top', functions=(fun_rev, fun_for))
    ax2.set_xlabel('Diameter [Å]')

# Plot the formation energies.
if plot_espec is True:
    fig, ax = plt.subplots(figsize=(10, 6))
    espec_list = [atoms.get_potential_energy()/len(atoms) for atoms in atoms_list]
    ax.scatter(natoms_list, espec_list, color='blue', s=30, alpha=0.5)
    ax.set_xlabel('Number of atoms [-]')
    ax.set_ylabel('Formation energy [eV/N$_{atoms}$]')
    ax.set_xlim(natoms_low, natoms_high)
    # Calculate diameter vs natoms function.
    fun_natoms = lambda diam, a_fit: a_fit*np.array(diam)**3
    a_fit = curve_fit(f=fun_natoms, xdata=diameter_list, ydata=natoms_list)[0][0]
    fun_for = lambda diam: a_fit*np.array(diam)**3
    fun_rev = lambda natoms: np.array(natoms/a_fit)**(1/3)
    ax2 = ax.secondary_xaxis('top', functions=(fun_rev, fun_for))
    ax2.set_xlabel('Diameter [Å]')

plt.show()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
