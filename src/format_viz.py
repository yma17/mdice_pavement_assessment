"""
File containing functions to write MRB and RRB results to shapefile
format, which can be easily visualized on the UI tool.
"""

import pandas as pd
import shapefile
import os, glob
import json


def make_directory(dir):
    try:
        os.mkdir(os.path.join("../output", dir))
    except FileExistsError:
        files = glob.glob(os.path.join("../output", dir) + "/*")
        for f in files:
            os.remove(f)


def str_to_2dlist(str):
    list = []
    pts = str[2:-2].split("), (")
    for pt in pts:
        list.append([float(x) for x in pt.split(',')])
    return list


def create_mrb_segment(df_mrb):
    make_directory("mrb_segment")
    w = shapefile.Writer("../output/mrb_segment/mrb_segment_results")
    w.shapeType = 8  # MultiPoint

    w.field('LRS_LINK', 'C', size=23)
    w.field('ROAD_NAME', 'C')
    w.field('FROM_ADDR', 'N')
    w.field('TO_ADDR', 'N')
    w.field('FROM_ZIP', 'N')
    w.field('TO_ZIP', 'N')
    w.field('LAST_RATING', 'N')  # most recent rating
    w.field('PREV_RATINGS', 'C', size=33)
    w.field('LEGALSYS', 'C')
    w.field('PAVINGS', 'C')
    w.field('LAST_PAVING', 'C')
    w.field('LENGTH', 'N', decimal=2)
    w.field('AADT', 'N')
    w.field('ADR', 'N')
    w.field('BUS_ROUTES', 'C')
    w.field('PASER_RATING_COMP', 'N')
    w.field('DECISION_SCORE', 'N', decimal=5)
    
    for _, row in df_mrb.iterrows():
        w.record(row['lrs_link'], row['road_name'], row['from_addr'],
            row['to_addr'], row['from_zip'], row['to_zip'], row['last_rating'],
            row['paser_ratings'], row['legalsys'], row['pavings'],
            row['last_paving'], row['length'], row['aadt'], row['adr'],
            row['routes'], row['paser_rating'], row['decision_score'])
        w.multipoint(str_to_2dlist(row["points"]))

    w.close()


def create_rrb_segment(df_rrb):
    make_directory("rrb_segment")
    w = shapefile.Writer("../output/rrb_segment/rrb_segment_results")
    w.shapeType = 8  # MultiPoint

    w.field('LRS_LINK', 'C', size=23)
    w.field('ROAD_NAME', 'C')
    w.field('FROM_ADDR', 'N')
    w.field('TO_ADDR', 'N')
    w.field('FROM_ZIP', 'N')
    w.field('TO_ZIP', 'N')
    w.field('LAST_RATING', 'N')  # most recent rating
    w.field('LEGALSYS', 'C')
    w.field('LENGTH', 'N', decimal=2)
    w.field('AADT', 'N')
    w.field('BENEFIT_SCORE', 'N', decimal=8)
    w.field('FAIRNESS_SCORE', 'N', decimal=8)
    w.field('QUALITY', 'N')
    w.field('CONDITION_REPAIR', 'N')  # if 1: condition low enough to repair
    w.field('MED_HOUSE_INCOME', 'N')  # estimated from nearby census blocks
    w.field('MED_PROPERTY_VAL', 'N')  # estimated from nearby census blocks
    w.field('POVERTY_RATE', 'N', decimal=5)  # est. from nearby census blocks
    
    for _, row in df_rrb.iterrows():
        w.record(row['LRS_LINK'], row['RDNAME'], row['FRADDR'], row['TOADDR'],
            row['ZIPL'], row['ZIPR'], row['LASTRATING'], row['LEGALSYS'],
            row['LENGTH'], row['AADT'], row['benefit_score'], row['fairness_score'],
            row['qualities'], row['condition'], row['median_household_income'],
            row['median_property_value'], row['poverty_rate'])

        mtp_o = row["geometry"][12:-1].split(",")
        mtp = []
        for x in mtp_o:
            mtp.append([float(y) for y in x.split(" ") if y])
        w.multipoint(mtp)

    w.close()


def create_rrb_blockgroup(df_rrb, df_ip, df_ipf, df_sgeoid):
    make_directory("rrb_blockgroup")
    w = shapefile.Writer("../output/rrb_blockgroup/rrb_blockgroup_results")
    w.shapeType = 5  # Polygon

    w.field('GEOID10', 'C', size=12)
    w.field('ALAND10', 'N')
    w.field('POP_ORIG', 'N')  # population before public asset inflation
    w.field('POP_ASSET', 'N')  # population before public asset inflation
    w.field('MEDIAN_GROSS_RENT', 'N')
    w.field('MEDIAN_HOUSEHOLD_INCOME', 'N')
    w.field('MEDIAN_PROPERTY_VALUE', 'N')
    w.field('RENT_BURDEN', 'N', decimal=1)
    w.field('POVERTY_RATE', 'N', decimal=5)
    w.field('PCT_RENTER_OCCUPIED', 'N', decimal=5)
    
    df_sgeoid = df_sgeoid.set_index('LRS_LINK')
    geoid_set = set()
    for _, row in df_rrb.iterrows():
        geoid = json.loads(df_sgeoid.loc[row['LRS_LINK'], 'geoid'].replace("'", "\""))
        for g in geoid.keys():
            geoid_set.add(g)
    if '' in geoid_set:
        geoid_set.remove('')
    geoid_list = list(geoid_set)
    df_ip = df_ip.astype({'GEOID10': 'str'})
    df_ipf = df_ipf.astype({'GEOID10': 'str'})
    df_ip = df_ip[df_ip['GEOID10'].isin(geoid_list)]
    df_ipf = df_ipf[df_ipf['GEOID10'].isin(geoid_list)]
    
    df_ip = df_ip.set_index('GEOID10')
    for _, row in df_ipf.iterrows():
        w.record(row['GEOID10'], row['ALAND10'],
            df_ip.loc[row['GEOID10'], 'population'], row['population'],
            row['median_gross_rent'], row['median_household_income'],
            row['median_property_value'], row['rent_burden'],
            row['poverty_rate'], row['pct_renter_occupied'])

        poly = []
        if row['geometry'][:7] == "POLYGON":
            polygon = row['geometry'][10:-2].split(",")
            for p in polygon:
                poly.append([float(y) for y in p.split(" ") if y])
            w.poly([poly])
        else:  # multipolygon
            polygons = row['geometry'][16:-3].split(")), ((")
            for polygon in polygons:
                poly_next = []
                for p in polygon.split(","):
                    poly_next.append([float(y) for y in p.split(" ") if y])
                poly.append(poly_next)
            w.poly(poly)

    w.close()


def format_viz():
    print("--- RUNNING DATA FORMATTER FOR VISUALIZATION (step 8) ---\n")

    # Load data.
    df_mrb = pd.read_csv("../output/mrb_results.csv")
    df_rrb = pd.read_csv("../output/rrb_results.csv")
    df_ip = pd.read_csv("../data/derived/df_impl_prep.csv")
    df_ipf = pd.read_csv("../data/derived/df_impl_prep_full.csv")
    df_sgeoid = pd.read_csv("../data/derived/segments_geoid.csv")

    #create_mrb_segment(df_mrb)
    #create_rrb_segment(df_rrb)
    create_rrb_blockgroup(df_rrb, df_ip, df_ipf, df_sgeoid)