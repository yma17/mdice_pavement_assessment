"""
File containing preprocessing code for AllRoads_Detroit data.
"""

from .prep_util import great_circle_distance

import shapefile
from shapely.geometry import Point, MultiPoint
from tqdm import tqdm
import geopandas as gpd
import numpy as np
import pickle

MEAN_EARTH_RADIUS_KM = 6371


def load_main_segments():
    ## Read data from shapefile.
    shapefile_path = "../data/AllRoads_Detroit/AllRoads_Detroit.shp"
    sf = shapefile.Reader(shapefile_path)
    #print("List of attributes")
    #print(sf.fields)

    ## Retrieve attribute values.
    ## Missing values are either 0 for type int, or '' for type str.
    shape_info = []
    #idx = 0
    recs = sf.shapeRecords()
    for sr in tqdm(recs):  # total: 38113 records.
        #if idx % 10 == 0:
        #    print("Retrieving attributes from record", idx)
        #idx += 1

        road_name = sr.record[1]  # str, missing for 1592 records.
        from_addr = sr.record[4]  # int, missing for 7673 records.
        to_addr = sr.record[5]  # int, missing for 7671 records.
        from_zip = sr.record[6]  # int, missing for 1096 records.
        to_zip = sr.record[7]  # int, missing for 1096 records.
        lrs_link = sr.record[32]  # str, present for all records
        length = sr.record[33]  # float, present for all records.
        legalsys = sr.record[40]  # str, missing for 1659 records.

        # FILTER: only select "State Trunkline", "City Major",
        # 	or "County Primary" roads, according to legalsys.
        if legalsys not in ["State Trunkline", "City Major", "County Primary"]:
            continue
        
        # Paser ratings by year, from 2009 to 2019 (in that order).
        # No ratings within this time span exist for 21115 records.
        paser_ratings = sr.record[76:66:-1] + [sr.record[96]]
        last_rating = sr.record[79]  # year of last rating.
        try:
            rating = paser_ratings[last_rating - 2009]
        except IndexError:
            rating = 0

        # Pavings by year, from 2018 to 2008 (in that order).
        # 29550 records indicate no pavement during this time period.
        pavings = sr.record[83:88] + sr.record[89:95]
        last_paving = None
        for i, paving in enumerate(pavings):
            if paving == "Yes":
                last_paving = 2018 - i
                break

        points = sr.shape.points  # list, present for all records.

        # Convert points from crs 2898 to 4326 (long/lat) format.
        geometry = [Point(p) for p in sr.shape.points]
        points = gpd.GeoDataFrame(crs="epsg:2898", geometry=geometry)
        points = points.to_crs(epsg=4326)
        points = [(row["geometry"].x, row["geometry"].y) for _, row in points.iterrows()]
        points_np = np.array(points)

        # Compute distances to equator and prime meridian for "center"
        # of road segment, in kilometers.
        mid_long = (points_np[:, 0].min() + points_np[:, 0].max()) / 2.0
        mid_lat = (points_np[:, 1].min() + points_np[:, 1].max()) / 2.0
        long_multiplier = 1 if mid_long >= 0 else -1
        lat_multiplier = 1 if mid_lat >= 0 else -1
        pm_dist = long_multiplier * great_circle_distance(
            mid_long, mid_lat, 0, mid_lat, MEAN_EARTH_RADIUS_KM
        )
        eq_dist = lat_multiplier * great_circle_distance(
            mid_long, mid_lat, mid_long, 0, MEAN_EARTH_RADIUS_KM
        )

        # Construct data structure.
        info = {
            "points": points_np,
            "attributes": {
                "road_name":road_name, "from_addr":from_addr, "to_addr":to_addr,
                "from_zip":from_zip, "to_zip":to_zip,
                "paser_ratings":paser_ratings, "last_rating":last_rating,
                "lrs_link":lrs_link, "legalsys":legalsys, "pavings":pavings,
                "last_paving":last_paving, "rating":rating, "length":length,
                "pm_dist_km":pm_dist, "eq_dist_km":eq_dist, "points": points
            }
        }
        shape_info.append(info)


    # Export to pickle file.
    with open('../data/derived/main_segments.pickle', 'wb') as outfile:
        pickle.dump(shape_info, outfile)