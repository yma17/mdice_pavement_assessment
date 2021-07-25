"""
File containing preprocessing functions for segment matcher.
"""

from .matcher_util import line_intersection, great_circle_distance, get_overlap

import numpy as np
import math
from scipy.interpolate import UnivariateSpline, interp1d

MEAN_EARTH_RADIUS_METERS = 6371009
MEAN_EARTH_RADIUS_12METERS = 6371009 / 12


def find_angle_category(theta):
    """
    For a certain angle measure theta, transform in range [0, pi),
    then place in one of the following categories:
    Category 0: [0, pi / 6)
             1: [pi / 6, pi / 3)
             2: [pi / 3, pi / 2)
             3: [pi / 2, 2 * pi / 3)
             4: [2 * pi / 3, 5 * pi / 6)
             5: [5 * pi / 6, pi)
    """
    return int((theta + math.pi) % math.pi / (math.pi / 6))


def convert_traj(traj):
    """
    Perform change of coordinates for a trajectory, converting from a
    list of (long, lat) points to a list of values of variable s that
    represents (approximate) arc length, that can model the functions:
        f: s -> long
        g: s -> lat
    Parameters: traj (_ x 2 numpy array)
    Returns: s_vals (_-length numpy array)
    Source: https://books.google.com/books?id=S7d1pjJHsRgC&pg=PA51#v=onepage&q&f=false
    """
    N = len(traj) - 1
    s_vals = [0]
    for i in range(N):
        next_s = s_vals[-1] + great_circle_distance(traj[i][0], traj[i][1],
                                traj[i + 1][0], traj[i + 1][1],
                                MEAN_EARTH_RADIUS_METERS)
        s_vals.append(next_s)
    
    return np.array(s_vals)


def spline_fit_traj(s_vals, traj, config):
    """
    Performs univariate spline fits on latitude and longitude
    of a trajectory, then retrieve interpolated trajectory.
    Parameters: traj (_ x 2 numpy array), s_vals (_-length numpy array)
    """
    long_vals, lat_vals = traj[:, 0], traj[:, 1]

    # offset to ensure difference between first original s value
    #	and first interpolated s value, is the same as difference
    #	between last original s value and last interpolated s value.
    offset = (s_vals[-1] % 5) / 2

    # Check if segment is sufficiently long enough for interpolation.
    # If not, take average point between first and last.
    if config["dist_between_pts"] <= s_vals[-1]:
        # Interpolated s values.
        s_interp_vals = np.array([i + offset for i in range(0,
                                        math.ceil(s_vals[-1]),
                                        config["dist_between_pts"])])

        # Perform interpolation.
        # If > 3 data points, use spline.
        # If = 3 data points, use quadratic.
        # Otherwise, use linear.
        if len(s_vals) > 3:
            long_spl = UnivariateSpline(s_vals, long_vals, s=0)  # s=0: no smoothing
            long_interp_vals = long_spl(s_interp_vals)
            lat_spl = UnivariateSpline(s_vals, lat_vals, s=0)  # s=0: no smoothing
            lat_interp_vals = lat_spl(s_interp_vals)
        else:
            kind = 'quadratic' if len(s_vals) == 3 else 'linear'
            long_lin = interp1d(s_vals, long_vals, kind=kind)
            long_interp_vals = long_lin(s_interp_vals)
            lat_lin = interp1d(s_vals, lat_vals, kind=kind)
            lat_interp_vals = lat_lin(s_interp_vals)
    else:
        long_interp_vals = np.array([long_vals[0], long_vals[-1]])
        lat_interp_vals = np.array([lat_vals[0], lat_vals[-1]])

    long_interp_vals = long_interp_vals.reshape(-1, 1)  # convert from 1d array
                                                        # to vertical 2d array
    lat_interp_vals = lat_interp_vals.reshape(-1, 1)
    return np.hstack((long_interp_vals, lat_interp_vals))


def compute_hash_for_traj(traj):
    """
    For a trajectory, compute a list of discrete coordinate/angle cells.
    Cells are 10 meters x 10 meters, and are identified in increasing
        order from west to east, and south to north.
    TODO: resolve bug at International Date Line.
    """
    pm_distances = []  # distances to prime meridian (0 degrees east)
    equator_distances = []  # distances to equator (0 degrees north)

    for pt in traj:
        long_multiplier = 1 if pt[0] >= 0 else -1
        lat_multiplier = 1 if pt[1] >= 0 else -1
        pm_distances.append(long_multiplier * great_circle_distance(
            pt[0], pt[1], 0, pt[1], MEAN_EARTH_RADIUS_12METERS
        ))
        equator_distances.append(lat_multiplier * great_circle_distance(
            pt[0], pt[1], pt[0], 0, MEAN_EARTH_RADIUS_12METERS
        ))

    hash_tuples = set()
    angles = []

    for i in range(traj.shape[0] - 1):
        #curr_pt = traj[i]
        #next_pt = traj[i + 1]
        #curr_eq_dist = equator_distances[i]
        #next_eq_dist = equator_distances[i + 1]
        #curr_pm_dist = pm_distances[i]
        #next_pm_dist = pm_distnaces[i + 1]

        theta = math.atan2(traj[i+1][1]-traj[i][1],traj[i+1][0]-traj[i][0])
        angle_category = find_angle_category(theta)
        angles.append(angle_category)

        # Add two initial hash tuples.
        hash_tuples.add((
            math.floor(pm_distances[i]),
            math.floor(equator_distances[i]),
            angle_category
        ))
        hash_tuples.add((
            math.floor(pm_distances[i + 1]),
            math.floor(equator_distances[i + 1]),
            angle_category
        ))

        # Compute lat/long distance lower/upper bounds.
        lat_dist_lower_bound = math.ceil(min(
            equator_distances[i], equator_distances[i + 1]
        ))
        lat_dist_upper_bound = math.ceil(max(
            equator_distances[i], equator_distances[i + 1]
        ))
        long_dist_lower_bound = math.ceil(min(
            pm_distances[i], pm_distances[i + 1]
        ))
        long_dist_upper_bound = math.ceil(max(
            pm_distances[i], pm_distances[i + 1]
        ))

        # For each cell border between lower and upper bounds
        # 	(for both latitude and longitude) inclusive
        # l1 is a line variable stored in the format:
        # 	[(long1, lat1), (long2, lat2)]
        l1 = [(pm_distances[i], equator_distances[i]),
              (pm_distances[i + 1], equator_distances[i + 1])]

        for lat_dist in range(lat_dist_lower_bound, lat_dist_upper_bound):
            # Find intersection between line represented by lat=lat_dist,
            #	and line segment between traj[i] and traj[i + 1].
            if long_dist_lower_bound != long_dist_upper_bound:
                l2 = [(long_dist_lower_bound, lat_dist),
                      (long_dist_upper_bound, lat_dist)]

                intersection_long = float(line_intersection(l1, l2)[0])

                # Add tuple for cell in between.
                hash_tuples.add((math.floor(intersection_long),
                    lat_dist,
                    angle_category
                ))
        #for
        for long_dist in range(long_dist_lower_bound, long_dist_upper_bound):
            if lat_dist_lower_bound != lat_dist_upper_bound:
                l2 = [(long_dist, lat_dist_lower_bound),
                      (long_dist, lat_dist_upper_bound)]

                intersection_lat = float(line_intersection(l1, l2)[1])

                hash_tuples.add((
                    long_dist,
                    math.floor(intersection_lat),
                    angle_category
                ))
        #for
    #for
    
    return hash_tuples, angles


def interpolate_dataset(data, config):
    """
    Interpolate trajectories from a dataset.
    Modifies: data, by adding the following key-value pairs to each dict obj:
        - "interp_points": interpolated trajectory (_ x 2 numpy array)
        - "ccd": cumulative chordal distance (float)
    """
    for i, d in enumerate(data):
        if i % 1000 == 0:
            print(i, "/", len(data))
        d_traj = d["points"]
        #print(d_traj)
        s_vals = convert_traj(d_traj)
        d_interp_traj = spline_fit_traj(s_vals, d_traj, config)
        #print(d_interp_traj)
        d["interp_points"] = d_interp_traj
        d["ccd"] = s_vals[-1]


def segment_trajectories(data, config):
    """
    Performs segmentation on a list of trajectories, such that
    each segment is at most config["seg_length"] meters long.
    Store segmented indices in data.
    """
    num_total_segments = 0
    for i, d in enumerate(data):
        if i % 1000 == 0:
            print(i, "/", len(data))
        num_segments = math.ceil(d["ccd"] / config["seg_length"])
        traj_num_points = d["interp_points"].shape[0]
        if traj_num_points == 1:
            print("NOPE", d["ccd"])
            exit(1)
        d["seg_idxs"] = []
        for j in range(num_segments):
            start = math.floor(j * traj_num_points / num_segments)
            end = math.floor((j + 1) * traj_num_points / num_segments)
            d["seg_idxs"].append((start, end))
        num_total_segments += num_segments

    return num_total_segments


def compute_hashes(data):
    for i, d in enumerate(data):
        if i % 1000 == 0:
            print(i, "/", len(data))
        d["seg_hashes"] = []
        d["seg_thetas"] = []
        for start, end in d["seg_idxs"]:
            h, a = compute_hash_for_traj(d["interp_points"][start:end])
            d["seg_hashes"].append(h)
            d["seg_thetas"].append(a)


def aggregate_bus_data(bus_data, config):
    """
    Aggregate bus data, so each road segment contains all routes that
    pass through them.
    """
    # Create seg_attributes for each bus data object.
    for d in bus_data:
        d["seg_attributes"] = {
            "adrs": [[d["attributes"]["adr"]]
                    for _ in range(len(d["seg_idxs"]))],
            "routes": [{d["attributes"]["route"]}
                    for _ in range(len(d["seg_idxs"]))]}
    
    # Check overlap between segments to aggregate adr + route data
    for i, d1 in enumerate(bus_data):
        for j, d2 in enumerate(bus_data):
            if i <= j:
                continue

            for k in range(len(d1["seg_idxs"])):
                for l in range(len(d2["seg_idxs"])):
                    if k <= l:
                        continue

                    ol = get_overlap(d1["seg_hashes"][k], d2["seg_hashes"][l])
                    if ol >= config["overlap_threshold"]:
                        d1["seg_attributes"]["adrs"][k].append(
                            d2["attributes"]["adr"])
                        d2["seg_attributes"]["adrs"][l].append(
                            d1["attributes"]["adr"])
                        d1["seg_attributes"]["routes"][k].add(
                            d2["attributes"]["route"])
                        d2["seg_attributes"]["routes"][l].add(
                            d1["attributes"]["route"])
    
    # Compute final adr for each segment.
    for d in bus_data:
        for i in range(len(d["seg_attributes"]["adrs"])):
            d["seg_attributes"]["adrs"][i]= sum(d["seg_attributes"]["adrs"][i])