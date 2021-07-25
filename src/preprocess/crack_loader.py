"""
File containing preprocessing code for crack data.
"""

# TODO(waitingl@umich.edu): import libraries here

import pickle
import os.path as osp
import pandas as pd
import geopandas as gpd


def load_crack():

    # Load crack data
    # csv file where each row: (image_url, # D00, # D10, # D20, # D40)
    print("Loading crack data")
    crack_data = pd.read_csv("../data/damage_detect/damage.csv")

    # Load MRB data, generated from segment_matcher.py
    print("Loading MRB data")
    mrb_data = pd.read_csv("../data/derived/mrb_data.csv")

    # Load shapefile for citywide road data
    print("Loading road data (shapefile)")
    roads = gpd.read_file(osp.join(
        "./../data/AllRoads_Detroit", "AllRoads_Detroit.shp")
    )

    # Load image to road data
    print("Loading image to road data")
    with open("../data/damage_detect/image2Road.data", "rb") as f:
        image2road = pickle.load(f)

    #####
    #Maps each crack to road segments.
    #####

    D00_mltplr = 1.0  # linear longitudinal crack
    D10_mltplr = 1.0  # linear lateral crack
    D20_mltplr = 2.0  # alligator crack
    D40_mltplr = 2.0  # rutting/bump/pothole/separation

    mrb_data["paser_rating"] = [10.0 for _ in range(len(mrb_data))]
    print("Running loop...")
    success_count = 0

    for _, crack_info in crack_data.iterrows():
        try:
            id, num_D00, num_D10, num_D20, num_D40 = crack_info
            image_key = id[:-4]
            road_index = image2road[image_key]
            lrs_link = roads.iloc[road_index]["LRS_LINK"]
            #print(lrs_link)
            #print(mrb_data[mrb_data['lrs_link'] == lrs_link])
            mrb_index = mrb_data[mrb_data['lrs_link'] == lrs_link].index.values[0]
            
            # Update paser rating
            paser_rating = mrb_data.iloc[mrb_index]["paser_rating"]
            paser_rating -= D00_mltplr * num_D00
            paser_rating -= D10_mltplr * num_D10
            paser_rating -= D20_mltplr * num_D20
            paser_rating -= D40_mltplr * num_D40
            paser_rating = max(1, int(paser_rating))
            mrb_data.at[mrb_index, "paser_rating"] = paser_rating

            success_count += 1
        except (KeyError, IndexError):
            pass

    mrb_data.to_csv("../data/derived/mrb_data_paser.csv", index=False)
    print(success_count, "out of", len(crack_data), "images matched to a road segment")