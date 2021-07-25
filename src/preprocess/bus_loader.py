"""
File containing preprocessing code for bus data.
"""

from .prep_util import clean_segment

import csv
import shapefile
import pickle
import numpy as np


def load_bus(borders):
    ## bus_info will be a list of objects for storing bus data. Each will contain:
    ##   Key: "route"  -> Value: <number of bus route>:
    ##   Key: "adr" -> Value: <annual daily ridership of route>
    ##   Key: "points" -> Value: <ordered list of coordinates for route>

    bus_info = []

    # Read bus ridership data.

    adr_dict = {}  # stores average daily ridership data
    route_numbers = set()  # unique route numbers
    data_months = 7  # March to September 2019

    monthly_ridership_filepath = "../data/bus_data/monthly_ridership.csv"

    with open(monthly_ridership_filepath, 'r') as mr_file:
        csvreader = csv.reader(mr_file, delimiter=',', quotechar='"')
        next(csvreader)
        for line in csvreader:
            route_number = int(line[1])
            route_ridership = int(line[3].replace(',',''))

            if route_number not in adr_dict.keys():
                adr_dict[route_number] = 0.0
                route_numbers.add(route_number)
            adr_dict[route_number] += float(route_ridership / data_months)


    # Read bus route data.

    shapefile_path = "../data/bus_data/d7a090b8-2691-4827-bca6-700822ae480f202042-1-h2r856.e2ch.shp"
    shape = shapefile.Reader(shapefile_path)

    for feature in shape.shapeRecords():
        points = feature.shape.points
        route_number = feature.record[3]
        #print(route_number)

        if route_number in route_numbers:  # ignore routes with no ridership data
            bus_obj = {"route": route_number,
                        "adr":adr_dict[route_number],
                        "points":points}

            bus_info.append(bus_obj)


    # Remove data outside of Detroit city limits.
    new_bus_info = []

    for info in bus_info:
        cleaned_route = clean_segment(info["points"], borders)

        for part in cleaned_route:
            new_bus_info.append({"points":np.array(part),
                                "attributes":{
                                    "adr":info["adr"],
                                    "route":info["route"]
                                }})


    # Save to pickle file
    with open('../data/derived/bus_segments.pickle', 'wb') as bus_data_file:
        pickle.dump(new_bus_info, bus_data_file)