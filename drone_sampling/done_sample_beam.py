import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import drone_sample_path as dsp
from scipy.spatial import KDTree
import release_points

###    This script is designed to the "cylinder" that the U10 camera + drone would see  ###



def plot_map(df,rig, release):


    plt.scatter(df['x'], df['z'], c = df["CH4_ppm_m"], cmap="viridis", label = "path integrated lazer sample", zorder = 2)
    cbar = plt.colorbar()
    cbar.set_label("tracer concentration ppm_m")

    plt.scatter(rig["Cx"], rig["Cz"], color='gray', linestyle='-', label='rig location', zorder=1, alpha=0.2)
    plt.xlabel("x"); plt.ylabel("z"); plt.title(f"Downwind slice of methane plume at distance: {abs(df["y"][0])}m \n with drone sampling path")
    txt = f"methane released from {release} at 0.01 kg/s"
    plt.figtext(0.5, 0.01, txt, ha='center')
    plt.legend(loc='upper left')

    plt.show()


# This function is intended to take all the rig coordinates and return the points of the downwind face of
# the rig. Basically, what the drone would actually bounce off of
def rig_sampling_face(rig_coords):

    #This is the boundary where all the y points will be further than
    threshhold = -15

    #slice data
    select_points = rig_coords[rig_coords["Cy"] < threshhold].reset_index(drop=True)


    return select_points


#This function finds the closest point in a straight line from the starting point.
#Starting point is where the drone would be in space
def find_end_point(start_point, rig_coords):

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


#This function samples the points of the cylinder along the path for the drone beam
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
    threshold = 3
    mask = distance < threshold

    return domain[mask]




def main():

    #the domain coordinates indicate where all the points are in space
    path = "/Users/lp27636/Desktop/OPENFOAM/project_1_CVX/vortex_shed/prod/6_ms_prod/domain.csv"
    domain_coords = pd.read_csv(path)
    domain_coords.rename(columns={"Points:0": "x", "Points:1": "y", "Points:2": "z", "CH4" : "tracer"}, inplace=True)
    release = release_points.check_for_matches(path.split("/"))

    #The coordinates indicate where the bounds of the CVX rig is located
    rig_coords = pd.read_csv("/Users/lp27636/Desktop/OPENFOAM/project_1_CVX/validation/super_fine/highwind_highrate_flare/rig.csv")
    rig_coords.rename(columns= {"Points:0": "Cx", "Points:1": "Cy", "Points:2": "Cz"}, inplace=True)
    rig_coords = rig_coords[["Cx","Cy","Cz"]]

    #Isolate Rig points to just the face of sampling
    rig_coords = rig_sampling_face(rig_coords)

    results_list = []
    #temporary start point - add loop later for all the different start points
    for start_point in dsp.main():
    #start_point = np.array([0, -100, 53])
        print("start_point:", start_point)

        #Get end point
        end_point_index = find_end_point(start_point, rig_coords)
        end_point = rig_coords.iloc[end_point_index].to_numpy()

        # Sample from start to end point and convert to a total molecule value
        beam_points = sample_points(start_point, end_point, domain_coords)
        #print(beam_points)


        #This section supported by gemini
        # 1. Define molecular weights of the individual gases (g/mol)
        MW_N2 = 28.0134
        MW_O2 = 31.999
        MW_CH4 = 16.04

        # Define the volume fractions of air (adjust if your OpenFOAM background differs)
        vol_frac_N2 = 0.79
        vol_frac_O2 = 0.21

        # Calculate the effective molecular weight of your air mixture
        MW_air = (vol_frac_N2 * MW_N2) + (vol_frac_O2 * MW_O2)

        # 2. Convert mass fraction ('tracer') to volume fraction (ppm_v)
        beam_points['tracer_ppm_v'] = beam_points['tracer'] * (MW_air / MW_CH4) * 1e6

        # 3. Calculate the path length of the cylinder (distance from start to end)
        path_length = np.linalg.norm(start_point - end_point)

        # 4. Get the average concentration of the sampled cylinder
        if not beam_points.empty:
            avg_ppm = beam_points['tracer_ppm_v'].mean()
        else:
            avg_ppm = 0.0  # Failsafe if the cylinder hits an empty region

        # 5. Calculate final ppm-m
        ppm_m = avg_ppm * path_length
        results_list.append([start_point[0],start_point[1], start_point[2], ppm_m])
        print(f"Path Length: {path_length:.2f} m | Average ppm: {avg_ppm:.2f} | ppm-m: {ppm_m:.2f}")

    results_df = pd.DataFrame(results_list, columns=["x", "y", "z", "CH4_ppm_m"])

    plot_map(results_df, rig_coords, release)



main()
