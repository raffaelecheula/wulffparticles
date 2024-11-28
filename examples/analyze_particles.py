# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

# Parameters.
natoms_low = 12000
natoms_high = 16000

# Ase Database.
db_name = 'Ni_new.db'
db_ase = connect(db_name)
selection = ""

# Get list of ase Atoms from ase database.
atoms_list = []
for id in [aa.id for aa in db_ase.select(selection=selection)]:
    atoms_row = db_ase.get(id=id)
    atoms = atoms_row.toatoms()
    atoms.info = atoms_row.data
    atoms.info.update(atoms_row.key_value_pairs)
    atoms_list.append(atoms)

atoms = atoms_list[0]
for ii, aa in enumerate(atoms):
    if atoms.info['ncoords'][ii] == 11:
        aa.symbol = "Zr"
    if atoms.info['ncoords'][ii] == 10:
        aa.symbol = "Mn"
    if atoms.info['ncoords'][ii] == 9:
        aa.symbol = "Cu"
    if atoms.info['ncoords'][ii] == 8:
        aa.symbol = "Au"
    if atoms.info['ncoords'][ii] == 7:
        aa.symbol = "Pt"
del atoms[atoms.info['ncoords'] == 12]
atoms.edit()
exit()

fig, ax = plt.subplots(figsize=(10, 6))

#natoms = 2000
#dfactor = 1e-3
#energy_list = [
#    atoms.get_potential_energy()*natoms/len(atoms) + 
#    natoms*dfactor*np.exp(len(atoms)/natoms)
#    for atoms in atoms_list
#]
deltamu = +0.10
energy_list = [atoms.get_potential_energy()-deltamu*len(atoms) for atoms in atoms_list]
natoms_list = [atoms.info["diameter"] for atoms in atoms_list]
ax.scatter(natoms_list, energy_list, color='blue', s=30, alpha=0.5)
ax.set_xlabel('Number of atoms [-]')
ax.set_ylabel('Formation energy [eV]')
ax.set_xlim(natoms_low, natoms_high)
#ax.set_ylim(-100, 1000)
plt.show()

temperature = 800 # [K]
kB = 8.61733e-5 # [eV/K]
probabilities = np.exp(-np.array(energy_list-np.min(energy_list))/(temperature*kB))
probabilities /= np.sum(probabilities)
print(np.max(probabilities))

ncoord_average = 0
for pp in np.argsort(-probabilities):
    if probabilities[pp] < 1e-3:
        break
    print(atoms_list[pp].info['ncoords'])
#atoms = sorted(atoms_list, key=lambda atoms: len(atoms))[-1]
#atoms.edit()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
