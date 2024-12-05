# **wulffparticles** 

A Python library for studying ensembles of nanoparticles using modified Wulff constructions. This library provides tools to compute equilibrium and non-equilibrium shapes of nanoparticles based on surface energies and visualize their geometry in 2D and 3D.

---

## **Features**
- Compute equilibrium and non-equilibrium shapes of nanoparticles via the modified Wulff construction.
- Support for custom surface energy values and anisotropic systems.
- Identification of the active sites of the nanoparticles.
- Visualization of nanoparticle shapes and the active sites in 3D.

---

## **Installation**
To install `wulffparticles`, clone the repository and install the requirements:
```bash
git clone https://github.com/raffaelecheula/wulffparticles.git
cd wulffparticles
pip install -e .
```

---

## **Getting Started**

### **Basic Example**
Here's a quick example of how to compute and visualize a nanoparticle shape:

```python
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

# Atoms bulk.
atoms_bulk = bulk('Ni', cubic=True)
# Reference surface energies.
surface_energies = {
    (1, 0, 0): 1.08,
    (1, 1, 0): 1.10,
    (1, 1, 1): 1.05,
    (2, 1, 1): 1.10,
}
# Get particle.
particle = AsymmetricParticle(
    surface_energies=surface_energies,
    primitive_structure=atoms_bulk,
    natoms=20000,
    asymm_multiplier=1.0,
)
# Get atoms.
atoms = particle.get_shifted_atoms(center_shift=None)
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
write_particle_and_atoms(particle=particle, atoms=atoms, filename="particle.png")
```

---

## **File Structure**
- `wulffparticles/`: Core module containing the implementation.
- `examples/`: Example scripts for typical use cases.

---

## **Contributing**
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](https://github.com/raffaelecheula/wulffparticles/LICENSE) file for details.

---

## **Contact**
For questions or suggestions, please open an issue on the [GitHub page](https://github.com/raffaelecheula/wulffparticles/issues).

--- 
