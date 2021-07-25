"""
Retrieves census ACS data at block level, saves into .csv file.
"""

import requests
import json
import pandas as pd
import numpy as np


def acs_read():
    # Request an API key here: https://api.census.gov/data/key_signup.html
    API_key = 'b3d819fd58bbd0e5b2f96f5f9d8905a6fc26c009'

    ################################
    #### relevant acs variables ####
    ################################
    ## see data dictionary for ACS dataset
    ## https://api.census.gov/data/2018/acs/acs5/variables.html
    acs_variables = ['GEO_ID', 'B01003_001E',
                    'B17010_002E', 'B17010_001E',
                    'B25003_003E', 'B25003_001E',
                    'B25064_001E', 'B19013_001E',
                    'B25077_001E', 'B25071_001E',
                    'B02001_002E', 'B02001_003E',
                    'B02001_004E', 'B02001_005E',
                    'B02001_006E', 'B02001_007E',
                    'B02001_008E']

    ## keep these columns
    acs_dic = {'GEO_ID': 'GEOID10',
                'B01003_001E':'population',
                'B02001_002E':'pop_white',
            'B02001_003E':'pop_black_af_am',
            'B02001_004E':'pop_am_indian_ak_native',
            'B02001_005E':'pop_asian',
            'B02001_006E':'pop_nat_hawaiian_pac_islander',
            'B02001_007E':'pop_some_other_race',
            'B02001_008E':'pop_two_or_more_races',
            'B25064_001E':'median_gross_rent',
            'B19013_001E':'median_household_income',
            'B25077_001E':'median_property_value',
            'B25071_001E':'rent_burden'}

    ## drop these columns (they only will be used to derive other quantities)
    acs_drop = ['B17010_002E', 'B17010_001E',
                'B25003_003E', 'B25003_001E']

    acs_variables_string = ','.join(acs_variables)

    ## employ acs API to generate a catalog (zip code level)
    # EXAMPLE: url = "https://api.census.gov/data/2018/acs/acs5?get=B17001_002E&for=zip%20code%20tabulation%20area:37064,37061,37060"

    ######################################
    #### Retrieving info at the tract ####
    ######################################
    # state 26 is Michigan
    ## see https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697
    ## see https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict13.txt
    #####################################################################################################
    # url = "https://api.census.gov/data/2018/acs/acs5?get="+acs_variables_string+"&for=tract:*&in=state:26&key="+API_key
    # response = requests.get(url)

    # parsed = json.loads(response.text)
    # data = np.array(parsed)
    # df = pd.DataFrame(data=data[1:,0:], columns=data[0,0:])  # [values + 1st row as the column names]

    # del data, parsed, response

    # # save and load them, this allows strings become numbers
    # df.to_csv('../../data/derived/acs_4399_FL_tract_level.csv', index=False)
    # df = pd.read_csv('../../data/derived/acs_4399_FL_tract_level.csv')

    # # compute some derived qunatities
    # df['poverty_rate'] = df['B17010_002E'] / df['B17010_001E']
    # df['pct_renter_occupied'] = df['B25003_003E'] / df['B25003_001E']

    # df.rename(columns=acs_dic, inplace=True)
    # df.drop(columns=acs_drop, inplace=True)
    #####################################################################################################
    # save the final catalog in a csv format
    #df.to_csv('../../data/derived/acs_4399_FL_tract_level.csv', index=False)

    ############################################
    #### Retrieving info at the block level ####
    ############################################
    ## check out Michigan counties here
    # https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697
    county_ids = ['%03d'%(163) ]

    df_a = []

    for cid in  county_ids:
        #print(cid)
        url = "https://api.census.gov/data/2018/acs/acs5?get="+\
                acs_variables_string+"&for=block%20group:*&in=state:26%20county:"+cid+"&key="+API_key
        response = requests.get(url)

        try:
            parsed = json.loads(response.text)
            data = np.array(parsed)
            df_a += [pd.DataFrame(data=data[1:,0:], columns=data[0,0:])] # [values + 1st row as the column names]
        except:
            print('   Could not retrieve county %s'%cid)

    df = pd.concat(df_a)
    del data, parsed, response

    # save and load them, this allows strings become numbers
    df.to_csv('../data/derived/block_group_tabular.csv', index=False)
    df = pd.read_csv('../data/derived/block_group_tabular.csv')

    # compute some derived qunatities
    df['poverty_rate'] = df['B17010_002E'] / df['B17010_001E']
    df['pct_renter_occupied'] = df['B25003_003E'] / df['B25003_001E']

    df.rename(columns=acs_dic, inplace=True)
    df.drop(columns=acs_drop, inplace=True)

    # save the final catalog in a csv format
    df.to_csv('../data/derived/block_group_tabular.csv', index=False)


# ```
# How the derived quantities are computed.

# population
#     Total population
#     2011-2018 uses 2015 5-year ACS B01003_001E

# poverty-rate
#     % of the population with income in the past 26 months below the poverty level
#     2011-2018 divides B17010_002E by B17010_001E in 2015 5-year ACS

# pct-renter-occupied
#     NOTE: This is not based off of the interpolated renter-occupied-households variable
#     % of occupied housing units that are renter-occupied
#     2011-2018 divides B25003_003E by B25003_001E in 2015 5-year ACS

# median-gross-rent
#     Median gross rent
#     2011-2018 uses 2015 5-year ACS B25064_001E

# median-household-income
#     Median household income
#     2011-2018 uses 2015 5-year ACS B19013_001E

# median-property-value
#     Median property value
#     2011-2018 uses 2015 5-year ACS B25077_001E

# rent-burden
#     Median gross rent as a percentage of household income, max is 50% representing >= 50%
#     2011-2018 uses 2015 5-year ACS B25071_001E
#