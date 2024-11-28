# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from wulffpack.core import BaseParticle

# -------------------------------------------------------------------------------------
# GET FACET TYPES
# -------------------------------------------------------------------------------------

def get_facet_types(
    atoms: Atoms,
    particle: BaseParticle,
    primitive_structure: Atoms,
    symprec: float = 1e-5,
    epsi: float = 1e-5,
):
    """Get the facet types for each surface atom."""
    from wulffpack.core.geometry import get_symmetries, get_standardized_structure
    from scipy.spatial import ConvexHull
    # Get asymmetric facet types from convex hull.
    hull = ConvexHull(atoms.positions, qhull_options="Qc")
    hkl_dict = {}
    facet_types = [[] for ii in range(len(atoms))]
    for equation in np.unique(hull.equations, axis=0):
        # Get hkl indices from plane equation.
        hkl = equation[:3]/np.min([abs(ii) for ii in equation[:3] if abs(ii) > epsi])
        hkl = tuple([int(round(ii)) for ii in hkl])
        # Get facet type.
        distances = np.dot(atoms.positions, equation[:3])
        for ii in np.where(np.abs(distances-equation[3]) < epsi)[0]:
            if len(facet_types[ii]) == 0 or hkl not in facet_types[ii]:
                facet_types[ii].append(hkl)
    # Substitute facet types with symmetrically equivalent hkl indices.
    standardized_structure = get_standardized_structure(
        structure=primitive_structure,
        symprec=symprec,
    )
    symmetries = get_symmetries(
        structure=standardized_structure,
        symprec=symprec,
    )
    for ii in range(len(facet_types)):
        for jj, hkl in enumerate(facet_types[ii]):
            if hkl not in hkl_dict:
                hkl_dict.update(get_parent_hkl_dict(hkl=hkl, symmetries=symmetries))
            facet_types[ii][jj] = hkl_dict[hkl]
        facet_types[ii] = sorted(facet_types[ii])
    return facet_types

# -------------------------------------------------------------------------------------
# GET PARENT HKL DICT
# -------------------------------------------------------------------------------------

def get_parent_hkl_dict(hkl, symmetries, hkl_dict={}):
    """Get a dictionary of symmatrically equivalent hkl indices."""
    hkl_tuple = []
    for rotation in symmetries:
        hkl_tuple.append(tuple([int(ii) for ii in np.dot(rotation, hkl)]))
    hkl_tuple = sorted(hkl_tuple, key=lambda x: (x[0], x[1], x[2]))
    for hkl in hkl_tuple:
        hkl_dict[hkl] = hkl_tuple[-1]
    return hkl_dict

# -------------------------------------------------------------------------------------
# GET FACET TYPES
# -------------------------------------------------------------------------------------

#def get_facet_types(atoms, particle):
#    """Get the facet types."""
#    facet_types = [[] for ii in range(len(atoms))]
#    for form in particle.forms:
#        hkl = ",".join([str(jj) for jj in form.parent_miller_indices])
#        for facet in form.facets:
#            distances = np.dot(atoms.positions, facet.original_normal)
#            for ii in np.where(distances > np.max(distances)-0.1)[0]:
#                facet_types[ii].append(hkl)
#    return facet_types

# -------------------------------------------------------------------------------------
# WRITE FACET TYPES
# -------------------------------------------------------------------------------------

def write_facet_types(
    atoms,
    facet_types,
    filename="atoms.png",
    scale=500,
    maxwidth=1000,
    radii=0.9,
    rotation="-90x,-30y,+45x",
    **kwargs,
):
    """Write the facet types."""
    colors = [[0,0,0] for ii in range(len(atoms))]
    for ii, types_list in enumerate(facet_types):
        if len(types_list) == 0:
            colors[ii] = [1,1,1]
        else:
            for type_jj in types_list:
                colors[ii] += np.array([float(kk) for kk in type_jj])
            colors[ii] /= np.linalg.norm(colors[ii])
            for jj in range(len(types_list)-1):
                colors[ii] = np.average([colors[ii], [0,0,1]], axis=0)
    # Write the atoms.
    atoms.write(
        filename=filename,
        scale=scale,
        maxwidth=maxwidth,
        radii=radii,
        colors=colors,
        rotation=rotation,
        **kwargs,
    )
    
# -------------------------------------------------------------------------------------
# SHOW FACET TYPES
# -------------------------------------------------------------------------------------

def show_facet_types(atoms, facet_types, show=True):
    """Show the facet types."""
    import matplotlib.pyplot as plt
    from ase.data import covalent_radii
    # Get radii and colors.
    radii = covalent_radii[atoms.numbers]
    # Prepare the figure.
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(6,6), dpi=100)
    fig.patch.set_facecolor("white")
    # Add the atoms.
    ax.scatter(
        xs=atoms.positions[:,0],
        ys=atoms.positions[:,1],
        zs=atoms.positions[:,2],
        color="w",
        alpha=1,
        edgecolors="k",
        linewidth=1,
        s=30*radii,
    )
    ax.view_init(elev=90, azim=0, roll=0)
    ax.grid(False)
    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')
    # Show the plot.    
    if show:
        plt.show()
    return ax

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
