# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------
# WRITE ATOMS AND SITES HKL
# -------------------------------------------------------------------------------------

def write_atoms_and_sites_hkl(
    atoms: Atoms,
    sites_hkl: list,
    filename: str = "atoms.png",
    scale: int = 500,
    maxwidth: int = 1000,
    radii: float = 0.9,
    rotation: str = "-90x,-30y,+45x",
    **kwargs,
):
    """Write an image with the hkl of the sites highlighted with different colors."""
    colors = [[0,0,0] for ii in range(len(atoms))]
    for ii, types_list in enumerate(sites_hkl):
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
# WRITE PARTICLE AND ATOMS
# -------------------------------------------------------------------------------------

def write_particle_and_atoms(
    particle,
    atoms,
    alpha: float = 0.65,
    linewidth: float = 0.5,
    colors: dict = None,
    legend: bool = True,
    atoms_color: str = "jmol",
    scale_particle: float = 0.90,
    scale_radii: float = 0.90,
    filename: str = "particle.png",
):
    """Write an image with the particle shape and the atoms inside."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from ase.data import covalent_radii
    # Plot atoms.
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = plt.axes(projection='3d')
    radii = covalent_radii[atoms.numbers]
    if atoms_color == "jmol":
        from ase.data.colors import jmol_colors
        atoms_color = jmol_colors[atoms.numbers]
    particle.make_plot(ax=ax, alpha=0, linewidth=linewidth, colors=colors)
    camera_zoom = (250 * scale_particle / np.max(np.abs(ax.get_w_lims()))) ** 2
    ax.scatter(
        xs=atoms.positions[:,0] * scale_particle,
        ys=atoms.positions[:,1] * scale_particle,
        zs=atoms.positions[:,2] * scale_particle,
        color=atoms_color,
        alpha=1,
        edgecolors="k",
        linewidth=1,
        s=scale_radii * camera_zoom * radii,
    )
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    plt.axis('off')
    if legend:
        fig.legend(frameon=False)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    im1 = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    plt.close(fig)
    # Plot particle.
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = plt.axes(projection='3d')
    particle.make_plot(ax=ax, alpha=1, linewidth=linewidth, colors=colors)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    plt.axis('off')
    if legend:
        fig.legend(frameon=False)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    im2 = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    # Combine images.
    combined = (1 - alpha) * im1 / 255.0 + alpha * im2 / 255.0
    plt.figure()
    plt.axis("off")
    plt.imsave(filename, combined)
    plt.close()

# -------------------------------------------------------------------------------------
# PLOT PARTICLE SIZE DISTRIBUTION
# -------------------------------------------------------------------------------------

def plot_particle_size_distribution(
    diameter_mean: float,
    diameter_stddev: float,
    diameter_list: list,
    probabilities: list,
):
    """Plot the distribution of particles."""
    # Calculate the weighted KDE.
    from scipy.stats import norm, gaussian_kde
    kde = gaussian_kde(dataset=diameter_list, weights=probabilities)
    # Plot the distribution.
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
    return fig

# -------------------------------------------------------------------------------------
# PLOT SITES CONCENTRATIONS
# -------------------------------------------------------------------------------------

def plot_sites_concentrations(
    sites_conc: dict,
    plot_fractions: bool = False,
):
    """Plot the sites concentrations."""
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
    return fig

# -------------------------------------------------------------------------------------
# PLOT SITES CONCENTRATIONS
# -------------------------------------------------------------------------------------

def plot_formation_energies(
    eform_list: list,
    natoms_list: list,
    diameter_list: list,
    natoms_low: float,
    natoms_high: float,
    specific_eform: bool = False,
    relative_eform: bool = False,
):
    """Plot the formation energies."""
    from scipy.optimize import curve_fit
    # Plot the formation energies.
    fig, ax = plt.subplots(figsize=(10, 6))
    if specific_eform is True:
        eform_list = [eform/natoms_list[ii] for ii, eform in enumerate(eform_list)]
    elif relative_eform is True:
        eform_list = [eform-min(eform_list) for eform in eform_list]
    ax.scatter(natoms_list, eform_list, color='blue', s=30, alpha=0.5)
    ax.set_xlabel('Number of atoms [-]')
    if specific_eform is True:
        ax.set_ylabel('Formation energy [eV/N$_{atoms}$]')
    elif relative_eform is True:
        ax.set_ylabel('Relative formation energy [eV]')
    else:
        ax.set_ylabel('Formation energy [eV]')
    ax.set_xlim(natoms_low, natoms_high)
    # Calculate diameter vs natoms function.
    fun_natoms = lambda diam, a_fit: a_fit*np.array(diam)**3
    a_fit = curve_fit(f=fun_natoms, xdata=diameter_list, ydata=natoms_list)[0][0]
    fun_for = lambda diam: a_fit*np.array(diam)**3
    fun_rev = lambda natoms: np.array(natoms/a_fit)**(1/3)
    ax2 = ax.secondary_xaxis('top', functions=(fun_rev, fun_for))
    ax2.set_xlabel('Diameter [Å]')
    return fig

# -------------------------------------------------------------------------------------
# WRITE PARTICLE DATA TO CSV
# -------------------------------------------------------------------------------------

def write_particle_data_to_csv(
    atoms_list: list,
    probabilities: list,
    replacements: dict = {},
    natoms_fun: callable = lambda atoms: atoms.info["natoms"],
    sites_fun: callable = lambda atoms, key: atoms.info[f"sites_distrib_{key}"],
    sites_quantity: str = "number",
    filename: str = "particles_data.csv",
):
    """Write the particle data to a csv file."""
    import pandas as pd
    from wulffparticles.analyses import (
        get_sites_concentrations,
        replace_facets,
        remove_self_interfaces,
    )
    sites_names = {"facets": {}, "edges": {}, "corners": {}}
    natoms_surf_list = []
    sites_distrib_list = []
    for ii, atoms in enumerate(atoms_list):
        sites_distrib_dict = {}
        for key in sites_names:
            sites_distrib = sites_fun(atoms=atoms, key=key)
            replace_facets(dictionary=sites_distrib, replacements=replacements)
            remove_self_interfaces(dictionary=sites_distrib)
            sites_distrib_dict[key] = sites_distrib
            for name in sites_distrib:
                sites_names[key][name] = None
            if key == "facets":
                natoms_surf_list.append(
                    sum([sites_distrib[name] for name in sites_distrib])
                )
        sites_distrib_list.append(sites_distrib_dict)
    data = []
    for ii, atoms in enumerate(atoms_list):
        natoms = natoms_fun(atoms=atoms)
        natoms_surf = natoms_surf_list[ii]
        sites_quantities = {}
        for key in sites_names:
            for name in sorted(sites_names[key]):
                sites_number = sites_distrib_list[ii][key].get(name, 0)
                if sites_quantity == "number":
                    sites_quantities[name] = sites_number
                elif sites_quantity == "concentration":
                    sites_quantities[name] = sites_number/natoms
                elif sites_quantity == "fraction":
                    sites_quantities[name] = sites_number/natoms_surf
        data_dict = {
            "id": atoms.info["id"],
            "natoms": natoms,
            "natoms_surf": natoms_surf,
            "probability": probabilities[ii],
        }
        data_dict.update(sites_quantities)
        data.append(data_dict)
    df = pd.DataFrame(data)
    for key in ["id", "natoms", "natoms_surf"]:
        df[key] = df[key].apply(lambda x: f"{x:06d}")
    df.to_csv(filename, index=False, float_format="%.10f")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
