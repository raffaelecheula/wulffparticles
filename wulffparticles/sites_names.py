# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from wulffpack.core import BaseParticle

# -------------------------------------------------------------------------------------
# GET SITES HKL
# -------------------------------------------------------------------------------------

def get_sites_hkl(
    atoms: Atoms,
    particle: BaseParticle,
    primitive_structure: Atoms,
    symprec: float = 1e-5,
    epsi: float = 1e-5,
) -> list:
    """Get the sites hkl for each surface atom."""
    from wulffpack.core.geometry import get_symmetries, get_standardized_structure
    from scipy.spatial import ConvexHull
    # Get asymmetric sites hkl from convex hull.
    hull = ConvexHull(atoms.positions, qhull_options="Qc")
    hkl_dict = {}
    sites_hkl = [[] for ii in range(len(atoms))]
    for equation in np.unique(hull.equations, axis=0):
        # Get hkl indices from plane equation.
        hkl = equation[:3]/np.min([abs(ii) for ii in equation[:3] if abs(ii) > epsi])
        hkl = tuple([int(round(ii)) for ii in hkl])
        # Get site hkl.
        distances = np.dot(atoms.positions, equation[:3])
        for ii in np.where(np.abs(distances-equation[3]) < epsi)[0]:
            if len(sites_hkl[ii]) == 0 or hkl not in sites_hkl[ii]:
                sites_hkl[ii].append(hkl)
    # Substitute sites hkl with symmetrically equivalent hkl indices.
    standardized_structure = get_standardized_structure(
        structure=primitive_structure,
        symprec=symprec,
    )
    symmetries = get_symmetries(
        structure=standardized_structure,
        symprec=symprec,
    )
    for ii in range(len(sites_hkl)):
        for jj, hkl in enumerate(sites_hkl[ii]):
            if hkl not in hkl_dict:
                hkl_dict.update(get_parent_hkl_dict(hkl=hkl, symmetries=symmetries))
            sites_hkl[ii][jj] = hkl_dict[hkl]
        sites_hkl[ii] = sorted(sites_hkl[ii])
    return sites_hkl

# -------------------------------------------------------------------------------------
# GET PARENT HKL DICT
# -------------------------------------------------------------------------------------

def get_parent_hkl_dict(
    hkl: tuple,
    symmetries: list,
    hkl_dict: dict = {},
) -> dict:
    """Get a dictionary of symmatrically equivalent hkl indices."""
    hkl_tuple = []
    for rotation in symmetries:
        hkl_tuple.append(tuple([int(ii) for ii in np.dot(rotation, hkl)]))
    hkl_tuple = sorted(hkl_tuple, key=lambda x: (x[0], x[1], x[2]))
    for hkl in hkl_tuple:
        hkl_dict[hkl] = hkl_tuple[-1]
    return hkl_dict

# -------------------------------------------------------------------------------------
# GET SITES DISTRIBUTION
# -------------------------------------------------------------------------------------

def get_sites_distribution(
    sites_hkl: list,
) -> dict:
    """Get the sites distribution."""
    sites_distrib = {}
    for hkl_list in sites_hkl:
        sites_distrib[tuple(hkl_list)] = sites_distrib.get(tuple(hkl_list), 0) + 1
    return sites_distrib

# -------------------------------------------------------------------------------------
# REDUCE SITES DISTRIBUTION
# -------------------------------------------------------------------------------------

def reduce_sites_distribution(
    sites_distrib: dict,
    n_hkl_sites: int = 1,
) -> dict:
    """Reduce the sites distribution to only tuples with length n_hkl_sites."""
    from itertools import combinations
    sites_distrib_red = {}
    for hkl_tuple in sites_distrib:
        if len(hkl_tuple) == n_hkl_sites:
            sites_distrib_red[hkl_tuple] = (
                sites_distrib_red.get(hkl_tuple, 0) + sites_distrib[hkl_tuple]
            )
        elif len(hkl_tuple) > n_hkl_sites:
            for hkl_red in combinations(hkl_tuple, n_hkl_sites):
                sites_distrib_red[hkl_red] = (
                    sites_distrib_red.get(hkl_red, 0) + sites_distrib[hkl_tuple]
                )
    return dict(sorted(sites_distrib_red.items()))

# -------------------------------------------------------------------------------------
# GET SITES DISTRIBUTION TYPES
# -------------------------------------------------------------------------------------

def get_sites_distribution_types(
    sites_hkl: list,
    sitetype_dict: dict = {"facets": 1, "edges": 2, "corners": 3},
):
    """Get the sites distribution for each type of site."""
    # Get sites distribution.
    sites_distrib = get_sites_distribution(sites_hkl=sites_hkl)
    # Reduce sites distribution.
    sites_distrib_dict = {}
    for key in sitetype_dict:
        sites_distrib_dict[key] = reduce_sites_distribution(
            sites_distrib=sites_distrib,
            n_hkl_sites=sitetype_dict[key],
        )
    return sites_distrib_dict

# -------------------------------------------------------------------------------------
# GET SITENAME
# -------------------------------------------------------------------------------------

def get_sitename(
    hkl_tuple: tuple,
    hkl_sep : str = "",
) -> str:
    """Get the sitename from a tuple of hkl indices."""
    return "+".join([hkl_sep.join([str(ii) for ii in hkl]) for hkl in hkl_tuple])

# -------------------------------------------------------------------------------------
# GET SITENAMES DISTRIBUTION
# -------------------------------------------------------------------------------------

def get_sitenames_distribution(
    sites_distrib: dict,
) -> dict:
    """Get the sitenames distribution."""
    sitenames_distrib_dict = {}
    for hkl_tuple in sites_distrib:
        sitename = get_sitename(hkl_tuple=hkl_tuple)
        sitenames_distrib_dict[sitename] = sites_distrib[hkl_tuple]
    return sitenames_distrib_dict

# -------------------------------------------------------------------------------------
# GET SITENAMES DISTRIBUTION TYPES
# -------------------------------------------------------------------------------------

def get_sitenames_distribution_types(
    sites_hkl: list,
    sitetype_dict: dict = {"facets": 1, "edges": 2, "corners": 3},
):
    """Get the sitenames distribution for each type of site."""
    sites_distrib_dict = get_sites_distribution_types(
        sites_hkl=sites_hkl,
        sitetype_dict=sitetype_dict,
    )
    sitenames_distrib_dict = {}
    for key in sitetype_dict:
        sitenames_distrib_dict[key] = get_sitenames_distribution(
            sites_distrib=sites_distrib_dict[key]
        )
    return sitenames_distrib_dict

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
