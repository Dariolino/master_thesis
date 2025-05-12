#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tu Mar 19 2025

@author: Dario DF

This script is for chromatic topological data analysis (TDA) using the Chromatic
TDA library. Its application is shown for the analysis of embryonic spatial
transcriptomics data, specifically the neural tube-somite-notochord interface.
"""
# %% import libraries
import chromatic_tda as chro
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import persim # for additional persistent images

# set directory
input_dir = "/Users/dario/Desktop/Lundeberg_thesis/visium_analysis/data/"

# load csv file generated from R script
data = pd.read_csv(input_dir + "se_res2_A_coords_clusters.csv")

# %% define keys and auxilliary functions
selected_keys = ["kernel", "relative", "cokernel", "sub_complex", "image",
                "complex"]

def get_persistent_diagrams(data, cluster1, cluster2, cluster3 = None,
                            show_dim_1_persistent_images = False,
                            filter_dim = None):
        """
        This function gets the persistent diagrams for the given clusters
        
        Parameters:
        - data (pandas.df): dataframe with columns "x", "y", and "cluster"
        - cluster1, cluster2 (int): cluster labels for comparison
        - cluster3 (int, optional): third cluster (if provided). If None, only
                                             cluster1 and cluster2 are used
        Plots:
        - Persistent diagrams in respect to given clusters
        """

        if cluster3 is not None: # often results in wide merge error due to dense data points
            # filter data for the given clusters
            data_clusters = data[(data["cluster"] == cluster1) | (data["cluster"] == cluster2) | (data["cluster"] == cluster3)]
            
            # get coordinates and labels
            points = data_clusters[["x", "y"]].values.astype(np.float64)
            labels = data_clusters["cluster"].tolist()
            
            # compute chromatic alpha complex and get simplicial complex
            chro_alpha = chro.ChromaticAlphaComplex(points, labels,
                                                    point_perturbation=1e-4)
            simplicial_complex = chro_alpha.get_simplicial_complex(
                                                     sub_complex="bi-chromatic",
                                                     full_complex="all",
                                                     relative="mono-chromatic") # putting relative = NONE results in different proportioning
    
        else: # for two colors/clusters
            # filter data for the given clusters
            data_clusters = data[(data["cluster"] == cluster1) | (data["cluster"] == cluster2)]
            
            # get coordinates and labels
            points = data_clusters[["x", "y"]].values.astype(np.float64)
            labels = data_clusters["cluster"].tolist()
            
            # compute chromatic alpha complex and get simplicial complex
            chro_alpha = chro.ChromaticAlphaComplex(points, labels,
                                                    point_perturbation=1e-4)
            simplicial_complex = chro_alpha.get_simplicial_complex(sub_complex = str(cluster1))

            if filter_dim is not None:
                filtered_complex = {key: simplicial_complex.bars(key, filter_dim) for key in selected_keys}
                chro.plot_six_pack(filtered_complex)
            else:
                chro.plot_six_pack(simplicial_complex)

        # get persistent diagrams (only dimension 1) --> easier interpretation
        if show_dim_1_persistent_images: # includes vectorization for downstream ML pipelines 

            # extract all relevant values for dimension 1 and compute max range
            bars_values = [simplicial_complex.bars(key)[1] for key in selected_keys]
            max_range = max(max(max(vals)) for vals in bars_values)

            persistence_imager = persim.PersistenceImager( 
                  birth_range = (0, max_range), pers_range = (0, max_range),
                  pixel_size = 0.05,  weight = lambda x, y: y ** 2,
                  weight_params = {}, # I increased the weighting to the exponent to highlight features of high persistence
                  kernel_params = {"sigma": ((0.2,0), (0,0.2))} 
            )

            diagrams = simplicial_complex.diagrams_list([["kernel", 1], 
                                                         ["relative", 1],
                                                         ["cokernel", 1],
                                                         ["sub_complex", 1],
                                                         ["image", 1],
                                                         ["complex", 1]])
            images = persistence_imager.transform(diagrams)

            fig, axs = plt.subplots(2, 3, figsize=(16, 12),
                                    gridspec_kw={'width_ratios': [1, 1, 1]})  
            axs = axs.flatten()  # flatten for easy iteration

            # assign each image and its corresponding title from selected_keys
            for ax, im, key in zip(axs, images, selected_keys):
                persistence_imager.plot_image(im, ax=ax)  # plot persistence image
                ax.set_title(key, fontsize=19)  # set subplot title
                
                # remove individual axis labels
                ax.set_xlabel("")  
                ax.set_ylabel("")

            # adjust subplot spacing to shift them to the right**
            plt.subplots_adjust(left=0.06, bottom=0.06, right=0.95, top=0.9,
                                wspace=0.1, hspace=0.1)

            # place the y-label into the new white space
            fig.text(0.02, 0.5, 'persistence', va='center', rotation='vertical',
                    fontsize=20)  # Move far left
            fig.text(0.5, 0.04, 'birth', ha='center', fontsize=20)  # x-axis label is fine
            plt.show()

        return None


def scatter_with_radius(ax, x, y, radius, **scatter_kwargs):
    """
    Plots a scatter plot where the marker size corresponds to a given radius in data units

    Parameters:
    - ax (matplotlib.axes.Axes): matplotlib axis to plot on
    - x, y (lists of floats): coordinates of points
    - radius (float): desired radius in data units
    - scatter_kwargs: additional arguments for plt.scatter()
    """
    # get figure DPI and axis size in pixels
    fig = ax.get_figure()
    ax_lim = ax.get_xlim()  # get x-axis limits
    ax_size = ax.get_window_extent().width  # width in pixels

    # compute scaling factor: convert from data units to points 
    data_range = ax_lim[1] - ax_lim[0]  # width of x-axis in data units
    pixels_per_data_unit = ax_size / data_range  # conversion factor

    # compute marker size
    marker_size = ((2 * radius * pixels_per_data_unit * 72. / fig.dpi) ** 2)

    # plot scatter
    ax.scatter(x, y, s=marker_size, **scatter_kwargs) # edgecolors="black"

def get_scatter_plot(data, cluster1, cluster2 = None, radius = 0.5,
                     color1 = "#1f4e79", color2 = "orange"):
    """
    Plots a scatter plot of the given clusters with a given radius

    Parameters:
    - data (pandas.df): dataframe with columns "x", "y", and "cluster"
    - cluster1, cluster2 (int): cluster labels for comparison
    - radius (float): radius of the scatter points in data units
    """

    # filter data for the given clusters
    data_clusters = data[(data["cluster"] == cluster1)] 

    if cluster2 is not None:
        data_clusters = data[(data["cluster"] == cluster1) | (data["cluster"] == cluster2)]

    # get x and y coordinates
    points = data_clusters[["x", "y"]].values.astype(np.float64)

    # define color based on cluster
    colors = np.where(data_clusters["cluster"] == cluster1, color1, color2)

    fig, ax = plt.subplots(dpi=141)

    # set axis limits (adjust based on your data)
    ax.set_xlim(min(points[:, 0]) - 10, max(points[:, 0]) + 10)
    ax.set_ylim(min(points[:, 1]) - 10, max(points[:, 1]) + 10)
    ax.set_aspect(1)  # Keep aspect ratio square

    # plot points using our custom scatter function
    scatter_with_radius(ax, points[:, 0], points[:, 1], radius=radius, c=colors,
                        edgecolors = "none", alpha = 0.5) #, marker="s", better to have none for edge clors so it matches what we expect exactly
    ax.set_title(f"Point cloud at radius {radius} - cluster {cluster1}",
                fontsize=12) 
    if cluster2 is not None:
        ax.set_title(f"Point cloud at radius {radius} - cluster {cluster1} and {cluster2}",
                     fontsize=12) 

    ax.set_xlabel("x coordinate", fontsize=11)  
    ax.set_ylabel("y coordinate", fontsize=11) 
    plt.show()

def filter_bars(bars_dict, birth_range=(0, float('inf')),     
                death_range=(0, float('inf')),
                persistence_range=(0, float('inf')), outer_keys=None, dims=None):  # outer keys and dims have to be in a list, even if it only contain sone objects
    """
    Filters a nested dictionary of bars based on birth, death, and persistence ranges.
    Also filters by outer keys and dimensions if specified.

    Parameters:
    - bars_dict (dict): original nested dictionary
    - birth_range (tuple): range for birth values (min, max)
    - death_range (tuple): range for death values (min, max)
    - persistence_range (tuple): range for persistence values (min, max)
    - outer_keys (list): list of outer keys to filter by (e.g., ["kernel", "relative"])
    - dims (list): list of dimensions to filter by

    Returns:
    - filtered_dict (dict): filtered nested dictionary
    """

    filtered_dict = {}

    # filter by outer keys if specified
    if outer_keys is not None:
        bars_dict = {key: bars_dict[key] for key in outer_keys if key in bars_dict}

    # filter by dimensions if specificied 
    if dims is not None:
        bars_dict = {
            key: {dim: inner_dict[dim] for dim in dims if dim in inner_dict}
            for key, inner_dict in bars_dict.items()
        }

    # unpack ranges
    birth_min, birth_max = birth_range
    death_min, death_max = death_range
    pers_min, pers_max = persistence_range

    # filter the bars based on the specified ranges
    for outer_key, inner_dict in bars_dict.items():
        filtered_inner = {}
        for dim, pairs in inner_dict.items():
            filtered_pairs = [
                (x, y) for (x, y) in pairs
                if birth_min <= x <= birth_max and
                   death_min <= y <= death_max and
                   pers_min <= (y - x) <= pers_max
            ]
            if filtered_pairs:
                filtered_inner[dim] = filtered_pairs
        if filtered_inner:
            filtered_dict[outer_key] = filtered_inner

    return filtered_dict

def find_pairs_with_matching_coords(bars_dict, coords, recursive=False): # caveat - it is possible that there ar false postives - so always check manually # function deos not work with H0 zero yet because then all H0 poitns are included
    """
    Filters bars_dict to include bars where either x or y from the input coords
    appear as birth or death. If recursive is True, expands to all bars connected
    by shared coordinates. Ensures all six required outer keys are present.
    
    Parameters:
    - bars_dict (dict): Original nested dictionary
    - coords (tuple): A pair (x, y) to match exactly
    - recursive (bool): Whether to expand to all connected coordinates
        
    Returns:
    - matched_dict (dict): Filtered nested dictionary
    """
    matched_dict = {}
    seen_coords = set()
    to_check = set(coords)

    # check if the input coords are in the bars_dict
    while len(to_check) > 0:
        current_coord = to_check.pop()
        seen_coords.add(current_coord)

        # check if the current coordinate is in any of the bars
        for outer_key, dim_dict in bars_dict.items():
            for dim, pairs in dim_dict.items():
                for birth, death in pairs: 
                    if birth == current_coord or death == current_coord: 
                        if outer_key not in matched_dict:   
                            matched_dict[outer_key] = {}    
                        if dim not in matched_dict[outer_key]:
                            matched_dict[outer_key][dim] = []
                        if (birth, death) not in matched_dict[outer_key][dim]:
                            matched_dict[outer_key][dim].append((birth, death))

                            if recursive: 
                                if birth not in seen_coords:
                                    to_check.add(birth)
                                if death not in seen_coords:
                                    to_check.add(death)

    # ensure all required keys are present, even if empty
    required_keys = ['kernel', 'relative', 'cokernel', 'sub_complex', 'image',
                    'complex']
    for key in required_keys:
        if key not in matched_dict:
            matched_dict[key] = {}

    chro.plot_six_pack(matched_dict)
    plt.show()

    return matched_dict


# %% example usage get_persistent_diagrams
get_persistent_diagrams(data, 0, 2, show_dim_1_persistent_images = True) # filter_dim = 1 for H1

# %% example usage get_scatter_plot
get_scatter_plot(data, 0, 2, radius =0.5)

# %% example usage filter_bars
data_cl_0_and_2 = data[(data["cluster"] == 0) | (data["cluster"] == 2)]

# get x and y coordinates and labels
points = data_cl_0_and_2[["x", "y"]].values # .tolist()  
labels = data_cl_0_and_2["cluster"] #.tolist()

# create chromatic alpha complex
chro_alpha = chro.ChromaticAlphaComplex(points, labels, point_perturbation = 1e-4) # with that pertruabtion it is stable


simplicial_complex = chro_alpha.get_simplicial_complex(sub_complex="0") # we study inclusion of zero space into everyting 
filtered_output = filter_bars(simplicial_complex.bars_six_pack(),
                              outer_keys= ["kernel"], dims = [1],
                              birth_range = (0,2)) 
print(filtered_output)
# %% example usage find_pairs_with_matching_coords # can help with annotations
bars_dict_test = simplicial_complex.bars_six_pack()
find_pairs_with_matching_coords(bars_dict_test, (4.554126448468839, 4.656569249755981), recursive=False)
