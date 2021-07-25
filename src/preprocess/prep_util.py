"""
File containg utility functions for preprocessing.
"""

from shapely.geometry import Point
import math
import geopandas as gpd
import pandas as pd
import matplotlib as plt


def load_shape_file(filename, loc='../data/', show_plot=True, column='rating', to_dataframe=True):
    gdf = gpd.read_file(loc+filename)
    if show_plot:
        gdf.plot(column=column, cmap=None)
        plt.show()
    if to_dataframe:
        return pd.DataFrame(gdf)
    else:
        return gdf


def clean_segment(segment, border_info):
	"""Exclude points from segment not in Detroit city limits."""
	cleaned_segment = []
	currently_within_limits = False
	portion_begin = 0

	for i in range(len(segment)):
		this_pt = Point(segment[i][0], segment[i][1])
		is_inside_outer = border_info["outer"].contains(this_pt)
		is_outside_inner = not(border_info["inner"].contains(this_pt))
		
		if not(is_inside_outer and is_outside_inner):  # outside city limits
			if currently_within_limits and i - portion_begin > 1:
				cleaned_segment.append(segment[portion_begin:i])
				currently_within_limits = False
		elif not currently_within_limits:
			currently_within_limits = True
			portion_begin = i

	if currently_within_limits and i - portion_begin > 1:
		cleaned_segment.append(segment[portion_begin:])

	return cleaned_segment


def great_circle_distance(pt1_long, pt1_lat, pt2_long, pt2_lat, radius):
    """
    Uses Haversine formula to compute great circle distance.
    Source: https://en.wikipedia.org/wiki/Great-circle_distance
    """
    phi_1 = math.radians(pt1_lat)
    phi_2 = math.radians(pt2_lat)
    delta_phi = abs(phi_1 - phi_2)
    delta_lambda = abs(math.radians(pt1_long) - math.radians(pt2_long))

    sqrt_term1 = math.sin(delta_phi / 2) ** 2
    sqrt_term2 = math.cos(phi_1) * math.cos(phi_2)
    sqrt_term3 = math.sin(delta_lambda / 2) ** 2
    sqrt_term = math.sqrt(sqrt_term1 + sqrt_term2 * sqrt_term3)

    return 2 * radius * math.asin(sqrt_term)