"""
'Main' file for segment matching.
"""

from .preprocess import interpolate_dataset, segment_trajectories
from .preprocess import compute_hashes, aggregate_bus_data

from .run_problem import compute_limits, assign_trajectories
from .run_problem import duplicate_trajectories, segment_matcher_subproblem
from .run_problem import combine_assignments

import pickle
import json
import csv
from random import sample
#import matplotlib.pyplot as plt
from statistics import mean


def load_data():
    """Load data from pickle files."""
    with open("../data/derived/main_segments.pickle", "rb") as f:
        main_segments = pickle.load(f)
    with open("../data/derived/traffic_segments.pickle", "rb") as f:
        traffic_segments = pickle.load(f)
    with open("../data/derived/bus_segments.pickle", "rb") as f:
        bus_segments = pickle.load(f)
    with open("preprocess/segment_matcher/segment_matcher_config.json", "rb") as f:
        config = json.load(f)

    return main_segments, traffic_segments, bus_segments, config


# def plot_assignments(final_assignments, quality_data, traffic_data,
#                      bus_data, num_plots=25):
#     names = ["traffic", "bus"]
#     names_abbr = ["t", "b"]
#     colors = ["r", "g"]

#     indices = sample(final_assignments.keys(),
#                      min(num_plots, len(final_assignments)))
#     for x, (i, seg_i) in enumerate(indices):
#         plt.clf()

#         seg_begin, seg_end = quality_data[i]["seg_idxs"][seg_i]
#         q_pts = quality_data[i]["interp_points"][seg_begin:seg_end]
#         q_label = "quality " + str(seg_begin) + " " + str(seg_end)
#         plt.scatter(q_pts[:, 0], q_pts[:, 1], c='b', label=q_label)

#         for y, data in enumerate([traffic_data, bus_data]):
#             for j, seg_j in final_assignments[(i, seg_i)][names_abbr[y]]:
#                 seg_begin, seg_end = data[j]["seg_idxs"][seg_j]
#                 pts = data[j]["interp_points"][seg_begin:seg_end]
#                 label = names[y] + " " + str(seg_begin) + " " + str(seg_end)
#                 plt.scatter(pts[:, 0], pts[:, 1], c=colors[y], label=label)

#         plt.title("Sample plot " + str(x))
#         plt.legend("lower right")
#         plt.show()


def combine_attributes(final_assignments, quality_data,
                       traffic_data, bus_data):
    """
    Perform 'join' on datasets from final assignments, performing
    aggregation of attributes. Specifically, for a quality data point:
        - take mean of aadt across all segments/matches.
        - take mean of adr across all segments/matches.
        - combine bus routes.
        - join other attributes as normal.
    """
    output_data = {}
    for (q_orig_idx, _), tb_indices in final_assignments.items():
        output_data[q_orig_idx] = output_data.get(q_orig_idx, {})

        # Add quality attributes.
        for attr, value in quality_data[q_orig_idx]["attributes"].items():
            output_data[q_orig_idx][attr] = value
        # Add traffic attributes.
        for (t_orig_idx, _) in tb_indices["t"]:
            for attr, value in traffic_data[t_orig_idx]["attributes"].items():
                if attr == "aadt":
                    output_data[q_orig_idx][attr] = \
                        output_data[q_orig_idx].get(attr, [])
                    output_data[q_orig_idx][attr].append(value)
                    continue
                output_data[q_orig_idx][attr] = value
        # Add bus attributes (skip if no bus match).
        if "b" not in tb_indices:
            output_data[q_orig_idx]["adr"] = 0
            output_data[q_orig_idx]["routes"] = set()
            continue

        output_data[q_orig_idx]["adr"] = output_data[q_orig_idx].get("adr", [])
        output_data[q_orig_idx]["routes"] = \
            output_data[q_orig_idx].get("routes", set())
        for (b_orig_idx, b_seg_idx) in tb_indices["b"]:
            # Add aadt.
            output_data[q_orig_idx]["adr"].append(
                bus_data[b_orig_idx]["seg_attributes"]["adrs"][b_seg_idx])
            # Add routes.
            output_data[q_orig_idx]["routes"].update(
                bus_data[b_orig_idx]["seg_attributes"]["routes"][b_seg_idx])

    for _, data in output_data.items():
        data["aadt"] = mean(data["aadt"])
        data["adr"] = mean(data["adr"]) if len(data["adr"]) > 0 else 0.0

    return output_data


def output_to_files(output_data):
    """Output to csv and pickle files."""
    with open("../data/derived/mrb_data.pickle", "wb") as pickle_file:
        pickle.dump(output_data, pickle_file)

    with open("../data/derived/mrb_data.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        attributes = list(next(iter(output_data.values())).keys())
        writer.writerow(attributes)
        for _, data_pt in output_data.items():
            row = [data_pt[attr] for attr in attributes]
            writer.writerow(row)


def find_subproblem_of_index(idx, subproblem_array, key):
    r_q, c_q, found = None, None, False
    for i in range(len(subproblem_array)):
        for j in range(len(subproblem_array[0])):
            if idx in subproblem_array[i][j][key]:
                r_q, c_q = i, j
                found = True
            if found:
                break
        if found:
            break
    return r_q, c_q


def match_segments():  # 'main' function
    quality_data, traffic_data, bus_data, config = load_data()

    # Preprocess data.
    dataset_names = ["quality", "traffic", "bus"]
    for i, data in enumerate([quality_data, traffic_data, bus_data]):
        print("Performing interpolation:", dataset_names[i])
        interpolate_dataset(data, config)
        print("Done. Number of trajectories interpolated:", len(data))

        print("Performing segmentation:", dataset_names[i])
        num_segments = segment_trajectories(data, config)
        print("Done. Number of trajectories from segmentation:", num_segments)

        print("Computing hash data:", dataset_names[i])
        compute_hashes(data)
        print("Done. Number of trajectories hashed:", num_segments)

    # Aggregate bus data, so each road segment contains all routes that
    #  pass through them.
    print("Aggregating bus data")
    aggregate_bus_data(bus_data, config)
    print("Done aggregating bus data")

    # Set up subproblems.
    r, c = config["div_rows"], config["div_cols"]
    subproblem_array = [[{"q": [], "t": [], "b": [], "finished": False}
                        for _ in range(c)] for _ in range(r)]
    limits = compute_limits([quality_data, traffic_data, bus_data],
                            subproblem_array)
    print("Extrema for overall problem(min long, max long, min lat, max lat):")
    print(*[limits["long"][0], limits["long"][1],
            limits["lat"][0], limits["lat"][1]])
    for i in range(r):
        for j in range(c):
            print("Extrema for subproblem", i, j, ":")
            sub_limits = subproblem_array[i][j]["limits"]
            print(*[sub_limits["long"][0], sub_limits["long"][1],
                    sub_limits["lat"][0], sub_limits["lat"][1]])

    # Assign data for each subproblem.
    dataset_keys = ["q", "t", "b"]
    for i, data in enumerate([quality_data, traffic_data, bus_data]):
        print("Assigning", dataset_names[i], "trajectories to subproblems")
        assign_trajectories(data, subproblem_array, limits, dataset_keys[i])

    # Duplicate each segment in traffic and bus datasets for match
    # 	robustness.
    print("Duplicating trajectories for robustness")
    for i in range(1, 3):
        duplicate_trajectories(subproblem_array, dataset_keys[i])
    
    # Continuously assign segments until no more can be assigned
    it = 0
    final_assignments = {}
    while True:
        print("Assignment iteration:", it)
        print()
        it += 1

        # Find solution to each subproblem, then combine subproblem solutions
        print("Beginning subproblems")
        for i in range(r):
            for j in range(c):
                print("Performing subproblem", i, j)
                segment_matcher_subproblem(subproblem_array, i, j, quality_data,
                                        traffic_data, bus_data, config)
        print("Combining results from subproblems")

        iteration_assignments = combine_assignments(subproblem_array)
        if not iteration_assignments:
            print("No more assignments found. Stopping...")
            break
        final_assignments.update(iteration_assignments)

        # Remove segments from each dataset that have been assigned already
        for q_idx, tb_indices in iteration_assignments.items():
            r_q, c_q = find_subproblem_of_index(q_idx, subproblem_array, "q")
            subproblem_array[r_q][c_q]["q"].remove(q_idx)

    print("Done. Number of final assignments:", len(final_assignments))
    #plot_assignments(final_assignments, quality_data, traffic_data, bus_data)
    output_data = combine_attributes(final_assignments, quality_data,
                                     traffic_data, bus_data)

    print("Outputting to csv file")
    output_to_files(output_data)