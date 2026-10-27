"""Compare horizontal +y beams with the original minimum-angle beam method.

Run beside drone_sample_path.py using the same Python environment as before.
The two methods use identical cell-centre data, drone positions, and cylinder
radius. Each reading sums CH4 * rho * Volume over selected cells, giving methane
mass in kg. CH4 must be mass fraction, rho kg/m3, and Volume m3. Whole cells are
included by their centres; partial cell-cylinder intersections are not resolved.

Left: full rig, approximately first forward target within the x-z tolerance;
      no target/no samples are missing measurements, displayed as gray crosses.
Right: original Cy < -15 rig filter and minimum-angle endpoint selection;
       empty sample cylinders still produce zero, matching the original script.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

DOMAIN_CSV = Path("/Users/lp27636/Desktop/OPENFOAM/project_1_CVX/validation/rho_react_recieved_fine/centertop/highwind_highrate/domain_cells.csv")
RIG_CSV = Path("/Users/lp27636/Desktop/OPENFOAM/project_1_CVX/validation/super_fine/highwind_highrate_flare/rig.csv")
OUTPUT_PNG = Path(__file__).resolve().with_name("beam_comparison.png")
RIG_TARGET_TOLERANCE = 1.0  # metres; point-cloud search tolerance
BEAM_RADIUS = 3.0           # metres; same as the original script
SHOW_MISSING = True         # Gray crosses are NOT zero-concentration readings.


def find_horizontal_end_point(start_point, rig_coords, tolerance=RIG_TARGET_TOLERANCE):
    """Approximate a forward rig hit while keeping the beam exactly along +y.

    The CSV contains discrete points, not connected surface faces. The tolerance
    finds nearby target points; it is not the physical beam radius. An exact
    first-surface intersection requires the rig surface mesh.
    Return None when there is no matching target ahead of the drone.
    """
    start_point = np.asarray(start_point, dtype=float)
    lateral_distance = np.hypot(
        rig_coords["Cx"] - start_point[0],
        rig_coords["Cz"] - start_point[2],
    )
    candidates = rig_coords[
        (lateral_distance <= tolerance)
        & (rig_coords["Cy"] > start_point[1])
    ]
    if candidates.empty:
        return None

    end_y = candidates["Cy"].min()
    return np.array([start_point[0], end_y, start_point[2]], dtype=float)


def find_original_end_point(start_point, rig_coords):

    closeness_threshold = 1
    inrange = False

    #for testing purposes we will start with an arbitrary point
    direction = [0, 1, 0]

    # Convert dataframe to numpy array
    rig_coords_array = rig_coords[["Cx","Cy","Cz"]].to_numpy()

    #Vector subtraction to create vectors from starting point to the rig
    rig_vectors = rig_coords_array - start_point

    #Use dot product to calculate how misaligned each vector is to the normal vector
    dot_prod = np.dot(rig_vectors, direction)

    #Calculate magnitudes
    direction_norm = np.linalg.norm(direction)
    rig_coords_norm = np.linalg.norm(rig_vectors, axis = 1)

    cos_theta = dot_prod / (rig_coords_norm * direction_norm)
    degrees = np.degrees(np.arccos(cos_theta))

    min_degree_index = np.argmin(degrees)
    return min_degree_index


def sample_points(start_point, end_point, domain):

    #Later implement KDtree to speed up sampling algorithm
    #For now just slice front and back of points to avoid grabbing unnecessary points
    domain = domain[(domain["y"] <= end_point[1])] ; domain = domain[(domain["y"] >= start_point[1])].reset_index(drop=True)
    domain_array = domain[["x","y","z"]].to_numpy()

    ### Calculate distance from line to each point ##

    #find direction vector d
    d = (end_point - start_point)/np.linalg.norm(end_point - start_point)

    #define vector from each domain point to the start point
    v = domain_array - start_point

    # This gives distance between start point and the project of domain points on the line
    t = np.dot(v, d)

    #Projection P of domain points on the line, The np.newaxis reforms the array into a column vector
    P = start_point + t[:, np.newaxis] * d

    # Last calculate the distance from domain points to the perpendicular point on line, axis = 1 to perform row by row
    distance = np.linalg.norm(P - domain_array, axis=1)

    #slice the original domain with tracer values to the values within the radius of the line
    threshold = BEAM_RADIUS
    mask = distance < threshold

    return domain.loc[mask].copy()




def calculate_readings(start_points, rig, domain, method):


    """Retain all drone positions and mark missing readings explicitly."""
    if method not in ("horizontal", "original"):
        raise ValueError("method must be 'horizontal' or 'original'")
    target_rig = (rig if method == "horizontal" else
                  rig.loc[rig["Cy"] < -15].reset_index(drop=True))
    if method == "original" and target_rig.empty:
        raise ValueError("The original Cy < -15 filter found no rig points.")

    records = []
    for i, start in enumerate(np.asarray(start_points, dtype=float)):
        row = dict(x=start[0], y=start[1], z=start[2], CH4_mass_kg=np.nan,
                   end_x=np.nan, end_y=np.nan, end_z=np.nan,
                   sample_count=0, status="no_target")
        if method == "horizontal":
            end = find_horizontal_end_point(start, target_rig)
        else:
            index = find_original_end_point(start, target_rig)
            end = target_rig.iloc[index][["Cx", "Cy", "Cz"]].to_numpy(dtype=float)

        if end is not None:
            row.update(end_x=end[0], end_y=end[1], end_z=end[2])
            samples = sample_points(start, end, domain)
            row["sample_count"] = len(samples)
            if samples.empty:
                row["status"] = "no_samples"
                if method == "original":
                    row["CH4_mass_kg"] = 0.0  # Original script's empty-cylinder fallback.
            else:
                # Methane mass in the selected cells:
                # mass fraction × gas density (kg/m³) × cell volume (m³)
                if "rho" not in samples:
                    methane_mass_kg = (
                            samples["tracer"]
                            * 1.222
                            * samples["Volume"]
                    ).sum(skipna=False)

                else:

                    methane_mass_kg = (
                            samples["tracer"]
                            * samples["rho"]
                            * samples["Volume"]
                    ).sum(skipna=False)

                row["CH4_mass_kg"] = methane_mass_kg
                row["status"] = (
                    "ok" if np.isfinite(methane_mass_kg) else "invalid_value"
                )
        records.append(row)
        if (i + 1) % 50 == 0 or i + 1 == len(start_points):
            print(f"{method}: {i + 1}/{len(start_points)} positions processed", flush=True)
    return pd.DataFrame.from_records(records)


def plot_comparison(horizontal, original, rig, output_path=OUTPUT_PNG, show=True):
    """Use a shared color scale and identical axes for a fair visual comparison."""
    values = np.concatenate([horizontal["CH4_mass_kg"].to_numpy(),
                             original["CH4_mass_kg"].to_numpy()])
    finite = values[np.isfinite(values)]
    if finite.size:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            padding = max(abs(vmin) * 0.01, 1e-9)
            vmin, vmax = vmin - padding, vmax + padding
    else:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True,
                             constrained_layout=True)
    panels = [
        (axes[0], horizontal, rig, "Horizontal beams along +y", f"Full rig; {RIG_TARGET_TOLERANCE:g} m target tolerance"),
        (axes[1], original, rig.loc[rig["Cy"] < -15], "Original beam method", "Minimum-angle target; rig Cy < -15 m"),
    ]
    for ax, result, backdrop, title, subtitle in panels:
        ax.scatter(backdrop["Cx"], backdrop["Cz"], color="gray", s=10,
                   alpha=0.2, zorder=1, label="Rig points")
        valid = np.isfinite(result["CH4_mass_kg"])
        if SHOW_MISSING and (~valid).any():
            ax.scatter(result.loc[~valid, "x"], result.loc[~valid, "z"],
                       color="0.60", marker="x", s=22, linewidths=0.8,
                       zorder=2, label="No valid reading")
        ax.scatter(result.loc[valid, "x"], result.loc[valid, "z"],
                   c=result.loc[valid, "CH4_mass_kg"], cmap="viridis", norm=norm,
                   s=28, zorder=3, label="Beam reading")
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)
        ax.set_xlabel("x (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.12)

    # Include every drone position in the common axis limits even if missing
    # markers have been disabled. Include the full rig for context.
    for column, rig_column, setter in [("x", "Cx", axes[0].set_xlim),
                                       ("z", "Cz", axes[0].set_ylim)]:
        coords = np.concatenate([horizontal[column], original[column], rig[rig_column]])
        coords = coords[np.isfinite(coords)]
        if coords.size:
            pad = max(float(np.ptp(coords)) * 0.06, 1.0)
            setter(float(coords.min()) - pad, float(coords.max()) + pad)
    axes[0].set_ylabel("z (m)")
    if finite.size:
        fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=axes,
                     label="Methane mass in selected cells (kg)", shrink=0.82)
    title = "Drone beam sampling comparison"
    y_values = horizontal["y"].unique()
    if len(y_values) == 1:
        title += f" | drone dist from face = {abs(y_values[0] - rig["Cy"].min())} m"
    fig.suptitle(title, fontsize=15)
    fig.supxlabel(
        f"Both panels sum CH4 × rho × Volume over selected cell centres; radius = {BEAM_RADIUS:g} m.\n"
        "Gray crosses: missing readings. Original method retains zero for empty cylinders.",
        fontsize=9,
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison: {output_path}")
    if show:
        plt.show()
    return fig


def main():
    import drone_sample_path as dsp

    domain = pd.read_csv(DOMAIN_CSV).rename(columns={
        "Points:0": "x", "Points:1": "y", "Points:2": "z", "CH4": "tracer"})
    rig = pd.read_csv(RIG_CSV).rename(columns={
        "Points:0": "Cx", "Points:1": "Cy", "Points:2": "Cz"})
    rig = rig[["Cx", "Cy", "Cz"]]
    start_points = np.asarray(dsp.main(), dtype=float)
    if start_points.ndim != 2 or start_points.shape[1] != 3 or not len(start_points):
        raise ValueError("The drone path must contain at least one [x, y, z] position.")

    horizontal = calculate_readings(start_points, rig, domain, "horizontal")
    original = calculate_readings(start_points, rig, domain, "original")
    for name, result in [("Horizontal", horizontal), ("Original", original)]:
        print(f"{name} status counts: {result['status'].value_counts().to_dict()}")
    plot_comparison(horizontal, original, rig)
    return horizontal, original


if __name__ == "__main__":
    main()
