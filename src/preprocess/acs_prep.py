"""
File with functions to preprocess census ACS data.
"""

from .prep_util import load_shape_file
from .geoid_id import geoid_id, sel_geoid

import pandas as pd
import numpy as np
import pickle
from shapely.ops import unary_union, transform
from sklearn.ensemble import RandomForestRegressor
from pyproj import CRS, Transformer
from shapely.geometry import Point, Polygon, MultiPoint, LineString


def fill_missing_value(df_stats):
    """Use random forest regression to fill in missing values."""
    count = -1
    columns = []
    X = df_stats
    for i in df_stats.columns:
        if len(X[X[i].isnull()]) > 0:
            columns.append(i)
    X = df_stats.dropna(axis=0,how='any')
    X = X.drop(columns, axis = 1)
    predicted_l = {}
    for i in df_stats.columns:
        count += 1
        df_tmp = df_stats.dropna(axis=0,how='any')
        known = df_tmp.to_numpy()
        y = known[:, count]

        unknown = df_stats[df_stats[i].isnull()].to_numpy()
        if(np.size(unknown) == 0):
            continue
        X_in = df_stats[df_stats[i].isnull()].dropna(axis=1,how='any').to_numpy()

        rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
        rfr.fit(X,y)
        predicted = rfr.predict(X_in)
        predicted_l[i] = predicted
    for i in df_stats.columns:
        unknown = df_stats[df_stats[i].isnull()].to_numpy()
        if(np.size(unknown) == 0):
            continue
        df_stats.loc[(df_stats[i].isnull()), i] = predicted_l[i]
    return df_stats


def acs_prep():
    # Load census block geometry data
    df = load_shape_file(filename='tl_2018_26_tabblock10.shp', loc='../data/polygon/', show_plot=False, to_dataframe=True)
    
    df['GEOID10'] = df['GEOID10'].apply(int)
    df = df.sort_values(by = "GEOID10")

    # Filter by Wayne county
    df = df.loc[df['GEOID10'] >= 261630000000000]
    df = df.loc[df['GEOID10'] < 261640000000000]

    # Filter by Detroit city limits (first load border data)
    with open("../data/derived/detroit_borders.pickle", "rb") as f:
        b = pickle.load(f)
    ind_to_remove = []
    for index, row in df.iterrows():
        long, lat = float(row['INTPTLON10']), float(row['INTPTLAT10'])
        pt = Point(long, lat)
        if not(b["outer"].contains(pt) and not(b["inner"].contains(pt))):
            ind_to_remove.append(index)
    df = df.drop(ind_to_remove)

    select_list = ['GEOID10', 'ALAND10', 'geometry']
    df = df[select_list]

    # Aggregate data from block to block group
    df['GEOID10'] = df['GEOID10'].apply(str)
    df['GEOID10'] = df['GEOID10'].apply(sel_geoid,start=0, end=12)
    df_land = df.groupby('GEOID10').sum()  # sum over 'ALAND10', remove 'geometry'
    df_land = df_land.reset_index()
    df_shape = df[['GEOID10', 'geometry']].groupby('GEOID10').aggregate(unary_union)  # combine 'geometry' polygons
    df_shape = df_shape.reset_index()
    df = pd.merge(df_land, df_shape)

    # Load preprocessed ACS block group data
    dff = pd.read_csv('../data/derived/block_group_tabular.csv')
    dff['GEOID10'] = dff['GEOID10'].apply(sel_geoid,start=9, end=223)

    # Combine block group dataframes
    df = pd.merge(df, dff, left_on='GEOID10', right_on='GEOID10')
    #print(df.dtypes)
    geoid_id_agent = geoid_id(df)

    # Prepare attributes to compute later
    df_impl = df.copy()
    df_impl['seg_id'] = pd.Series([[]]*len(df_impl))
    df_impl['qualities'] = pd.Series([[]]*len(df_impl))
    df_impl['avg_quality'] = pd.Series([-1]*len(df_impl))
    df_impl['lengths'] = pd.Series([[]]*len(df_impl))
    df_impl.set_index('GEOID10', inplace=True)

    # Filter by attributes for df_impl
    df_impl = df_impl[['ALAND10', 'population', 'median_gross_rent',
        'median_household_income', 'median_property_value', 'rent_burden',
        'poverty_rate', 'pct_renter_occupied', 'avg_quality']]

    # Fill in missing values in dataframe
    for i in range(len(df_impl)):
        for j in range(len(df_impl.columns)):
            if (df_impl.iloc[i,j] < -666):
                df_impl.iloc[i,j] = np.nan
    df_impl = fill_missing_value(df_impl)
    
    df_impl['geometry'] = df['geometry'].values

    # Compute neighbour, neighbour population data.
    df['neighbour'] = pd.Series([[]]*len(df))
    df['neighbour_population'] = pd.Series([0]*len(df))
    n_index = df.columns.get_loc('neighbour')
    np_index = df.columns.get_loc('neighbour_population')
    for i in range(len(df)):
        neighbour = []
        neighbour_pop = 0
        polygon_src = df.iloc[i]['geometry']
        for j in range(len(df)):
            if i == j:
                continue
            polygon_tag = df.iloc[j]['geometry']
            if (polygon_src.touches(polygon_tag)):
                neighbour.append(df.iloc[j]['GEOID10'])
                neighbour_pop += df.iloc[j]['population']
        df.iat[i, n_index] = neighbour
        df.iat[i, np_index] = neighbour_pop
    df_impl['neighbour'] = df['neighbour'].values
    df_impl['neighbour_population'] = df['neighbour_population'].values

    # Save further preprocessed ACS data to file.
    df_impl.to_csv('../data/derived/df_impl_prep.csv')

    # Load all-city road segment data
    # Convert coordinate system to (long, lat)
    segments = load_shape_file(filename='AllRoads_Detroit.shp', loc='../data/AllRoads_Detroit/', show_plot=False, to_dataframe=True)
    coord_conversion = Transformer.from_crs(
        CRS.from_epsg(2898), CRS.from_epsg(4326), always_xy=True
    )
    geometry = segments['geometry']
    m_pts = [MultiPoint([pt for pt in list(line_str.coords)]) for line_str in geometry]
    new_linestrings = []
    for pt in m_pts:
        x_pts = list(transform(coord_conversion.transform, pt).geoms)
        new_coords = []
        for x_pt in x_pts:
            new_coords.extend(list(x_pt.coords))
        new_linestring = LineString(new_coords)
        new_linestrings.append(new_linestring)
    segments['geometry'] = pd.Series(new_linestrings)

    # Match road segments to census block groups.
    # Column items match GEOID10 to number of segment points in that block group.
    segments['geoid'] = segments['geometry'].apply(geoid_id_agent.find_geoid)

    # Save converted all-city road segment data to file.
    segments.to_csv('../data/derived/segments_geoid.csv', index=False)