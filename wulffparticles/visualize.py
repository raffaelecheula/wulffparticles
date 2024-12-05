# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms

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
    atoms_color: str = "w",
    atoms_scale: float = 0.97,
    atoms_radii: float = 120.0,
    filename: str = "particle.png",
):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from ase.data import covalent_radii
    # Plot atoms.
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = plt.axes(projection='3d')
    radii = covalent_radii[atoms.numbers]
    ax.scatter(
        xs=atoms.positions[:,0] * atoms_scale,
        ys=atoms.positions[:,1] * atoms_scale,
        zs=atoms.positions[:,2] * atoms_scale,
        color=atoms_color,
        alpha=1,
        edgecolors="k",
        linewidth=1,
        s=atoms_radii * radii,
    )
    particle.make_plot(ax=ax, alpha=0, linewidth=linewidth, colors=colors)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
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
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
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
# END
# -------------------------------------------------------------------------------------
