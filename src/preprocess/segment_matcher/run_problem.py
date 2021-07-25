"""
File containing assignment functions for segment matcher.
"""

from .matcher_util import great_circle_distance, intersect, get_overlap

import numpy as np
from scipy.optimize import linear_sum_assignment

MEAN_EARTH_RADIUS_METERS = 6371009


def get_overlap_theta(ang1, ang2):
    #TODO: this is currently a massive runtime bottleneck
    c = intersect(ang1, ang2)
    return len(c) / ((len(ang1) + len(ang2)) / 2.0)


def assignment_problem(subproblem_array, i, j, data1, data2,
                       data_key1, data_key2, config):
    """Performs the assignment problem for two datasets."""
    n = len(subproblem_array[i][j][data_key1])
    m = len(subproblem_array[i][j][data_key2])

    print("Performing subproblem", i, j, "with", data_key1, "and", data_key2)

    if n == 0 or m == 0:
        print("No trajectories found.")
        return None

    print(n, "trajectories in dataset", data_key1)
    print(m, "trajectories in dataset", data_key2)

    # Construct mapping from (orig_idx, seg_idx) to matrix index.
    idx_map = {data_key1: [None for _ in range(n)],
               data_key2: [None for _ in range(m)]}
    for data_key in (data_key1, data_key2):
        data_pt_idx = 0
        for orig_idx, seg_idx in subproblem_array[i][j][data_key]:
            idx_map[data_key][data_pt_idx] = (orig_idx, seg_idx)
            data_pt_idx += 1

    # Construct cost matrix.
    cost_matrix = np.zeros((n, m))
    overlap_matrix = np.zeros((n, m))
    for row_idx in range(n):
        for col_idx in range(m):
            orig_idx1, seg_idx1 = idx_map[data_key1][row_idx]
            orig_idx2, seg_idx2 = idx_map[data_key2][col_idx]

            overlap = get_overlap(
                data1[orig_idx1]["seg_hashes"][seg_idx1],
                data2[orig_idx2]["seg_hashes"][seg_idx2]
            )
            overlap_matrix[row_idx][col_idx] = overlap

            #begin1, end1 = data1[orig_idx1]["seg_idxs"][seg_idx1]
            #begin2, end2 = data2[orig_idx2]["seg_idxs"][seg_idx2]
            #theta_sim = get_overlap_theta(
            #    data1[orig_idx1]["interp_points"][begin1:end1],
            #    data2[orig_idx2]["interp_points"][begin2:end2]
            #)

            theta_sim = get_overlap_theta(data1[orig_idx1]["seg_thetas"][seg_idx1],
                      data2[orig_idx2]["seg_thetas"][seg_idx2])

            if overlap < config["overlap_threshold"]:
                cost_matrix[row_idx][col_idx] = -1
            elif theta_sim < config["theta_threshold"]:
                cost_matrix[row_idx][col_idx] = -1
            else:
                begin1, end1 = data1[orig_idx1]["seg_idxs"][seg_idx1]
                points_1 = data1[orig_idx1]["interp_points"][begin1:end1]
                begin2, end2 = data2[orig_idx2]["seg_idxs"][seg_idx2]
                points_2 = data2[orig_idx2]["interp_points"][begin2:end2]
                dists = []
                for pt1 in points_1:
                    for pt2 in points_2:
                        dist = great_circle_distance(pt1[0], pt1[1], pt2[0], pt2[1],
                                                     MEAN_EARTH_RADIUS_METERS)
                        dists.append(dist)
                cost_matrix[row_idx][col_idx] = min(dists)

    cost_matrix[cost_matrix == -1] = np.max(cost_matrix) + 1

    # Perform assignment problem using Hungarian algorithm.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    row_ind_data1 = [idx_map[data_key1][ind] for ind in row_ind]
    col_ind_data2 = [idx_map[data_key2][ind] for ind in col_ind]

    # Filter according to overlap threshold and
    #   min graph distance thresholds.
    assignments = {}
    for i in range(len(row_ind)):
        if overlap_matrix[row_ind[i], col_ind[i]] >= config["overlap_threshold"]:
            sel_ind1, sel_ind2 = row_ind_data1[i], col_ind_data2[i]
            assignments[sel_ind1] = assignments.get(sel_ind1, set())
            assignments[sel_ind1].add(sel_ind2)

    print("Filtered assignments:", len(assignments))

    return assignments


def compute_limits(all_data, subproblem_array):
    """Return long/lat ranges for overall problem + each subproblem."""

    # Compute extrema for problem over all data.
    min_long, max_long, min_lat, max_lat = 180.0, -180.0, 90.0, -90.0
    for data in all_data:
        for d in data:
            traj_min_coords = np.amin(d["interp_points"], axis=0)
            traj_max_coords = np.amax(d["interp_points"], axis=0)

            min_long = min(min_long, traj_min_coords[0])
            max_long = max(max_long, traj_max_coords[0])
            min_lat = min(min_lat, traj_min_coords[1])
            max_lat = max(max_lat, traj_max_coords[1])

    # Compute (approximate) limits for each subproblem.

    long_step_size = (max_long - min_long) / len(subproblem_array[0])
    lat_step_size = (max_lat - min_lat) / len(subproblem_array)

    for i in range(len(subproblem_array)):
        lat_lower_bound = min_lat + i * lat_step_size
        lat_upper_bound = lat_lower_bound + lat_step_size

        for j in range(len(subproblem_array[0])):
            long_lower_bound = min_long + j * long_step_size
            long_upper_bound = long_lower_bound + long_step_size

            subproblem_array[i][j]["limits"] = {
                "lat": (lat_lower_bound, lat_upper_bound),
                "long": (long_lower_bound, long_upper_bound)
            }

    return {"lat": (min_lat, max_lat), "long": (min_long, max_long)}


def assign_trajectories(data, subproblem_array, overall_limits, data_key):
    n_cols = len(subproblem_array[0])
    n_rows = len(subproblem_array)
    min_long = overall_limits["long"][0]
    max_long = overall_limits["long"][1]
    min_lat = overall_limits["lat"][0]
    max_lat = overall_limits["lat"][1]

    for i, d in enumerate(data):
        if i % 1000 == 0:
            print(i, "/", len(data))
        # Construct a set of cells for each segment, referenced by
        # 	their (row, col) index pairs, from a trajectory.
        for j, (start, end) in enumerate(d["seg_idxs"]):
            cells = set()
            for k in range(start, end):
                pt = d["interp_points"][k]
                col_idx = int(((pt[0] - min_long) /
                        (max_long - min_long)) // (1 / n_cols))
                row_idx = int(((pt[1] - min_lat) /
                        (max_lat - min_lat)) // (1 / n_rows))
                col_idx = min(col_idx, n_cols - 1)
                row_idx = min(row_idx, n_rows - 1)
                cells.add((row_idx, col_idx))

            # Store segments indexed by (orig_data_idx, seg_idx)
            for row_idx, col_idx in cells:
                subproblem_array[row_idx][col_idx][data_key].append((i, j))


def duplicate_trajectories(subproblem_array, data_key, k=1):
    """Duplicate subproblem arrays k times."""
    for i in range(len(subproblem_array)):
        for j in range(len(subproblem_array[0])):
            subproblem_array[i][j][data_key] += \
                k * subproblem_array[i][j][data_key]


def segment_matcher_subproblem(subproblem_array, i, j, quality_data,
                               traffic_data, bus_data, config):
    """Performs a specific subproblem."""
    if subproblem_array[i][j]["finished"]:
        print("Already finished problem", i, j, ".Skipping...")
        subproblem_array[i][j]["assignments"] = {}
        return

    qt_assignments = assignment_problem(subproblem_array, i, j, quality_data,
                                        traffic_data, "q", "t", config)
    qb_assignments = assignment_problem(subproblem_array, i, j, quality_data,
                                        bus_data, "q", "b", config)
    if not qt_assignments or not qb_assignments:
        print("Missing trajectories for problem", i, j, ".Skipping...")
        subproblem_array[i][j]["assignments"] = {}
        subproblem_array[i][j]["finished"] = True
        return
    
    # Combine results from quality-traffic and quality-bus problems.
    sub_assignments = {}
    data_keys = ["t", "b"]
    for k, assignments in enumerate([qt_assignments, qb_assignments]):
        for sel1_idx, sel2_idx_list in assignments.items():
            for sel2_idx in sel2_idx_list:
                sub_assignments[sel1_idx] = sub_assignments.get(
                    sel1_idx, {"t": set(), "b": set()})
                sub_assignments[sel1_idx][data_keys[k]].add(sel2_idx)

    subproblem_array[i][j]["assignments"] = sub_assignments


def combine_assignments(subproblem_array):
    """Combine assignment indices from all subproblems."""
    final_assignments = {}
    for i in range(len(subproblem_array)):
        for j in range(len(subproblem_array[0])):
            for q_idx, tb_indices in \
                subproblem_array[i][j]["assignments"].items():

                final_assignments[q_idx] = final_assignments.get(
                    q_idx, {"t": set(), "b": set()})
                final_assignments[q_idx]["t"].update(tb_indices["t"])
                final_assignments[q_idx]["b"].update(tb_indices["b"])

    # Remove assignments where traffic data is absent.
    # Bus data is allowed to be absent (not every road has a bus)
    ind_to_remove = []
    for ind, assignment in final_assignments.items():
        if not assignment["t"]:
            ind_to_remove.append(ind)
    for ind in ind_to_remove:
        final_assignments.pop(ind)

    return final_assignments