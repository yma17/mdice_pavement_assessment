"""
File containing preprocessing code for traffic volume data.
"""

from .prep_util import clean_segment

import shapefile
import pickle
import numpy as np


def load_traffic(borders):
    ## Read AADT data from shapefile.
    traffic_info = []
    shapefile_path = "../data/traffic_data/Traffic_Volume.shp"
    shape = shapefile.Reader(shapefile_path)

    ## traffic_info will consist of dictionary values, each representing an object
    ##   for storing data for a single traffic segment. each will contain:
    ##   Key: "aadt" -> Value: <annual average daily traffic of route>
    ##   Key: "points" -> Value: <ordered list of coordinates of segment path>

    for feature in shape.shapeRecords():
        feature_info = {}
        feature_info["points"] = feature.shape.points

        feature_info["attributes"] = {}
        feature_info["attributes"]["aadt"] = feature.record[1]

        traffic_info.append(feature_info)

    # print(len(traffic_info))


    ## Remove data outside of Detroit city limits.
    cleaned_traffic_info = []
    for i in range(len(traffic_info)):
        cleaned_segment = clean_segment(traffic_info[i]["points"], borders)

        ## If nothing in cleaned segment, none of original segment is in city limits
        ## If more than one segment in cleaned segment,
        ## parts of original segment not in city limits,
        ## since city limits were defined conservatively.
        
        for part in cleaned_segment:
            traffic_info[i]["points"] = np.array(part)
            cleaned_traffic_info.append(traffic_info[i])

    # print(len(cleaned_traffic_info))


    ## Save to pickle file
    with open('../data/derived/traffic_segments.pickle', 'wb') as traf_data_file:
        pickle.dump(cleaned_traffic_info, traf_data_file)