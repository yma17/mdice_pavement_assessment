"""
File containing preprocessing code for public asset data.
"""

from .prep_util import load_shape_file
from .geoid_id import geoid_id

import json
import pandas as pd
import geopandas as gpd
import math


def sel_geoid_first(strin):
    str_out = (list(strin.keys())[0])
    if(str_out == ''):
        return None
    return int(str_out)


def increase_pop(df, pop_inc, geoid):
    if math.isnan( geoid) != True:
        df.at[int(geoid), 'population'] += pop_inc


def load_assets():
    # Load data from filesystem.
    school = load_shape_file(
        filename='b7d6b1e3-0958-41bc-8853-1be68cf3031b2020328-1-p2x4qd.4pnx.shp',
        loc='../data/public_assets/school/', show_plot=False, to_dataframe=True
    )
    hospital = load_shape_file(filename='8f8147bc-6999-483c-a589-ec8b2ec831c22020329-1-1iv4wkx.mf0q.shp',
        loc='../data/public_assets/hospital/', show_plot=False, to_dataframe=True
    )
    grocery_stores = load_shape_file(filename='10fae949-9ad3-4cd3-996a-78239d78b516202047-1-17jq9hh.g4uj.shp',
        loc='../data/public_assets/grocery_stores/', show_plot=False, to_dataframe=True
    )
    with open('../config/public_asset_weights.json', 'r') as f:
        config = json.load(f)
    df = pd.read_csv("../data/derived/df_impl_prep.csv")

    school = school[['geometry']]
    grocery_stores = grocery_stores[['geometry']]

    df['GEOID10'] = df['GEOID10'].astype('object')
    df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
    #print(df.dtypes)

    # Assign public assets to geoids
    geoid_id_agent = geoid_id(df)
    school['geoid'] = school['geometry'].apply(geoid_id_agent.find_geoid)
    school['geoid'] = school['geoid'].apply(sel_geoid_first)
    hospital['geoid'] = hospital['geometry'].apply(geoid_id_agent.find_geoid)
    hospital['geoid'] = hospital['geoid'].apply(sel_geoid_first)
    grocery_stores['geoid'] = grocery_stores['geometry'].apply(geoid_id_agent.find_geoid)
    grocery_stores['geoid'] = grocery_stores['geoid'].apply(sel_geoid_first)

    df.set_index('GEOID10', inplace=True)
    df_copy = df.copy()

    # Increase population

    school['population'] = pd.Series([0]*len(school))
    for i in school.index:
        geoid = school.loc[i, 'geoid']
        if (math.isnan(geoid)):
            continue
        geoid = int(geoid)
        neighbour_pop = df_copy.at[geoid, 'neighbour_population']
        school.at[i, 'population'] = neighbour_pop * config['school_dau']
    
    hospital['population'] = pd.Series([0]*len(hospital))
    for i in hospital.index:
        num_beds = hospital.at[i,'BEDS']
        hospital.at[i, 'population'] = num_beds * config['hospital_dau']
    hospital = hospital[['geometry', 'population', 'BEDS', 'geoid']]
    
    grocery_stores['population'] = pd.Series([0]*len(grocery_stores))
    for i in grocery_stores.index:
        geoid = grocery_stores.loc[i, 'geoid']
        if (math.isnan(geoid)):
            continue
        geoid = int(geoid)
        neighbour_pop = df_copy.at[geoid, 'neighbour_population']
        grocery_stores.at[i, 'population'] = neighbour_pop * config['grocery_dau']

    for i in range(len(hospital)):
        geoid = hospital.loc[i,'geoid']
        pop_inc = hospital.loc[i,'population']
        increase_pop(df, pop_inc, geoid)

    for i in range(len(school)):
        geoid = school.loc[i,'geoid']
        pop_inc = school.loc[i,'population']
        increase_pop(df, pop_inc, geoid)

    for i in range(len(grocery_stores)):
        geoid = grocery_stores.loc[i,'geoid']
        pop_inc = grocery_stores.loc[i,'population']
        increase_pop(df, pop_inc, geoid)

    df.to_csv("../data/derived/df_impl_prep_full.csv")