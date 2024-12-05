# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np

# -------------------------------------------------------------------------------------
# GET PROBABILITIES
# -------------------------------------------------------------------------------------

def get_probabilities(
    eform_list: list, # [eV]
    temperature: float, # [K]
    kB: float = 8.61733e-5, # [eV/K]
) -> list:
    """Calculate the Boltzmann probabilities."""
    deltae = (np.array(eform_list)-np.min(eform_list))
    probabilities = np.exp(-deltae/(temperature*kB))
    probabilities = np.nan_to_num(probabilities, neginf=0.)
    probabilities /= np.sum(probabilities)
    return list(probabilities)

# -------------------------------------------------------------------------------------
# GET SITES CONCENTRATIONS
# -------------------------------------------------------------------------------------

def get_sites_concentrations(
    atoms_list: list,
    probabilities: list,
    natoms_fun: callable = lambda atoms: atoms.info["natoms"],
    sites_fun: callable = lambda atoms, key: atoms.info[f"sites_distrib_{key}"],
    probability_thr: float = 1e-3,
) -> dict:
    """Calaculate natoms distribution and sites distributions."""
    sites_conc = {"facets": {}, "edges": {}, "corners": {}}
    for pp in np.argsort(-np.array(probabilities)):
        if probabilities[pp] < probability_thr:
            break
        natoms = atoms_list[pp].info["natoms"]
        for key in sites_conc:
            sites_distrib = atoms_list[pp].info[f"sites_distrib_{key}"]
            for site in sites_distrib:
                sites_conc[key][site] = (
                    + sites_conc[key].get(site, 0)
                    + sites_distrib[site]/natoms * probabilities[pp]
                )
    return sites_conc

# -------------------------------------------------------------------------------------
# REPLACE FACETS
# -------------------------------------------------------------------------------------

def replace_facets(
    dictionary: dict,
    replacements: dict,
) -> dict:
    """Substitute facets in a dictionary."""
    site_del = []
    for site in dictionary.copy():
        if len([repl for repl in replacements if repl in site]) > 0:
            site_new = "+".join(sorted([
                replacements[ss] if ss in replacements else ss 
                for ss in site.split("+")
            ]))
            dictionary[site_new] = dictionary.get(site_new, 0) + dictionary[site]
            site_del.append(site)
    for site in site_del:
        del dictionary[site]
    return dictionary

# -------------------------------------------------------------------------------------
# REPLACE SELF INTERFACES
# -------------------------------------------------------------------------------------

def remove_self_interfaces(
    dictionary: dict,
) -> dict:
    """Remove interfaces between the same facets."""
    from collections import Counter
    site_del = []
    for site in dictionary:
        counter = Counter([ss for ss in site.split("+")])
        if max(dict(counter).values()) > 1:
            site_del.append(site)
    for site in site_del:
        del dictionary[site]
    return dictionary

# -------------------------------------------------------------------------------------
# CALCULATE NATOMS PARAMETER
# -------------------------------------------------------------------------------------

def calculate_natoms_parameter(
    deltamu_0: float, # [eV]
    natoms: int, # [-]
    atoms_list: list,
    get_eform_fun: callable,
    temperature: float, # [K]
    diameter_mean: float, # [Å]
    get_diameter_fun: callable = lambda atoms: atoms.info["diameter"],
) -> list:
    """Calculate the natoms parameter."""
    from scipy.optimize import minimize
    # Residuals function.
    def residuals(
        deltamu,
        natoms,
        atoms_list,
        get_eform_fun,
        temperature,
        diameter_mean,
        get_diameter_fun,
    ):
        eform_list = [
            float(get_eform_fun(atoms=atoms, natoms=natoms, deltamu=deltamu))
            for atoms in atoms_list
        ]
        probabilities = get_probabilities(
            eform_list=eform_list,
            temperature=temperature,
        )
        diameter_list = [
            float(get_diameter_fun(atoms=atoms))
            for atoms in atoms_list
        ]
        mean = np.average(diameter_list, weights=probabilities)
        return np.abs(diameter_mean-mean)
    # Minimize residuals.
    args = (
        natoms,
        atoms_list,
        get_eform_fun,
        temperature,
        diameter_mean,
        get_diameter_fun,
    )
    res = minimize(
        fun=residuals,
        x0=deltamu_0,
        args=args,
        method='Nelder-Mead',
    )
    return res.x[0]

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
