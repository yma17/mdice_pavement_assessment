# M-DICE Policy Proposal

This file specifies the details of the M-DICE policy proposal algorithm, and how to run the program.

## Setup / Library Requirements

The code has been tested and run in an Anaconda environment with Python 3.8, and uses the following libraries. To setup, first run these commands inside the root folder, in the following order:

`conda create -n mdice python=3.8`

`conda activate mdice`

`python -m pip install Shapely pyshp numpy`

`conda install geopandas`

`conda install -c conda-forge scikit-learn`

`cd src`

`python -m pip install -r requirements_0.txt` (only if you wish to run step 0 of the pipeline)

Then, go to the "Pavement Assessment" Google Drive folder (https://drive.google.com/drive/u/1/folders/1uu1kvQ03Ih8dqPvA_MIWDOBxCCqWUqSf) in order to retrieve the data for the repo. Under the `data` folder inside the Google Drive folder, there should be 7 `.zip` files, each named corresponding to a sub-folder of the `data/` folder inside the repo. Extract the files of each `.zip` file **directly** inside the corresponding folder in the repo (for example, should be `data/AllRoads_Detroit/AllRoads_Detroit.shp`, instead of `data/AllRoads_Detroit/AllRoads_Detroit/AllRoads_Detroit.shp`).

If you choose to run the model inference, you will have to place all desired road images under the `src/detector/images` folder.

## Pipeline and Methodology

At a high-level, the provided code does the following (in order):
* Train and utilize a model for detecting cracks from road image data
* Preprocess city-wide datasets for algorithm
* Run decision process algorithm, for both major roads and residential roads
* Perform fairness evaluation on residential roads
* Format major and residential results into shapefile format compatible with a UI tool to visualize them.

## Running the Program

Running `python policy_proposal.py` from the `src/` directory in order to run the end-to-end pipeline, either in whole or in part. By default, the command without any additional parameters will only run steps of the pipeline that come after the preprocessing (in other words, steps 6 through 8). An assumption here is that the fully preprocessed data has already been created; these data files have been included with the repo under `data/`.

Additional/specific steps of the pipeline can be run instead of the default behavior, if one wants to redo any part of the preprocessing steps and get different subsequent results from the decision process.

The command line parameter `-p` (or alternatively `--pipeline`) does this, specifying a sequence of step identifiers to follow. The step identifiers defined are as follows (see the 'Step Requirements' section for file requirements for each step):

0. Run crack detection model on images data under `src/detector/images`. [NOTE: the repo by default does not have any images under this directory. City-wide road image data must be manually placed here before running this step.] 
1. Run preprocessing for city-wide road segment, traffic, and bus datasets.
2. Combine step 1 datasets through coordinate sequence matching.
3. Run preprocessing for city-wide crack data, combine with step 2 output.
4. Run preprocessing for city-wide census data.
5. Run preprocessing for city-wide public asset data, combine with census data from Step 4.
6. Run major road benefit (MRB) decision process.
7. Run residential road benefit (RRB) decision process, including fairness evaluation.
8. Format MRB and RRB results into shapefile format, in order to be easily usable with UI visualization tool.

For example, the command `python policy_proposal.py -p 123` will run first step 1 of the pipeline, then step 2, and finally step 3.

## Miscellaneous Scripts

Under `src/misc`, there exists miscellaneous scripts seperate from the pipeline:
* `detroit_borders.py` creates a `detroit_borders.pickle` file under `data/derived` which delineate both the outer border and inner (Highland Park / Hamtramck) border of the city. This file is used for filtering in the preprocessing stage. It is already included as part of the repo, but the hardcoded values in the script can be altered to create a new border.

## Data and Step Requirements

NOTE: All preprocessed data produced from Steps 0 through 5 are already included in the repo. It is only necessary to run the decision process and final step, unless a change in data and/or data sources necesitates having to produce new preprocessed data.

For each of these steps in the pipeline, the following files must exist in the filesystem:

Step 0:
  * All image data, `.jpg`, under `src/detector/images`
  * (for the first time running this step only) An Internet connection, in order to download model weights from Google Drive.

Step 1:
  * A set of `.cpg`, `.dbf`, `.prj`, `.shp`, `.shp.xml`, and `.shx` files under `data/AllRoads_Detroit/`. Contains city-wide road segments with attributes such as address, previous ratings, etc.
  * A set of `.cpg`, `.csv`, `.dbf`, `.prj`, `.shp`, `.shp.xml`, and `.shx` files under `data/traffic_data/`. Contains city-wide road segments with traffic volume attribute.
  * A set of `.cpg`, `.dbf`, `.prj`, `.shp`, and `.shx` files, under `data/bus_data/`. Contains city-wide road segments representing bus routes.
  * `monthly_ridership.csv`, derived from `Historical Ridership_UM Project.xlsx`, under `data/bus_data/`. Contains city-wide bus ridership data.
  * `detroit_borders.pickle`, under `data/derived/`. Contains borders of Detroit city.

Step 2:
  * `main_segments.pickle`, `traffic_segments.pickle`, and `bus_segments.pickle`, under `data/derived/`. Contains (initally) preprocessed city-wide road, traffic, and bus data, respectively. Produced from Step 1.
  * `segment_matcher_config.json` under `preprocess/segment_matcher`. Contains parameters for segment matching preprocessing step. Default parameters come with the repo.

Step 3: [NOTE: as of 7/25/21, the repo does not yet contain the full code for this step]
  * `damage.csv` under `data/damage_detect/`. Contains crack statistics for each image of cracks in the city. Produced from Step 0.
  * Aforementioned all-city road data under `data/AllRoads_Detroit`.
  * `mrb_data.csv` under `data/derived/`. Contains info on city-wide road segments after passing through data preprocessing and combination. Produced from Step 2.
  * `image2Road.data` under `data/damage_detect/`. Contains mapping of image keys to indices in all-city road data.

Step 4:
  * An internet connection (for using API to read raw census data). The API key is hardcoded.
  * A set of `.cpg`, `.dbf`, `.prj`, `.shp`, `.shp.iso.xml`, `.shp.ea.iso.xml`, and `.shx` files under `data/polygon/`. Contains census block data, which includes those for Detroit city.
  * Aforementioned Detroit city border data under `data/derived/`.
  * Aforementioned all-city road data under `data/AllRoads_Detroit`.

Step 5:
  * `df_impl_prep.csv` under `data/derived/`. Contains preprocessed ACS census data with filled in missing values from random forest regression. Produced from Step 4.
  * School, hospital, and grocery store shapefile datasets under `data/public_assets/`.
  * `public_asset_weights.json` under `config/`. Contains parameter weights for different public assets.

Step 6:
  * `mrb_data_paser.csv` under `data/derived/`. Contains fully preprocessed major road data. Produced from Step 3.
  * `mrb_config.json` under `config/`. Contains parameters for major road decision algorithm.

Step 7:
  * Aforementioned all-city road data under `data/AllRoads_Detroit`.
  * `rrb_config.json` under `config/`. Contains parameters for residential road decision algorithm.
  * `segments_geoid.csv` under `data/derived/`. Road segment data with geometry and geoid attributes. Produced from Step 4.
  * `df_impl_prep_full.csv` under `data/derived/`. Contains fully preprocessed census data. Produced from Step 5.

Step 8:
  * `mrb_results.csv` under `output/`. Decision results of major road process. Produced from Step 6.
  * `rrb_results.csv` under `output/`. Decision results of residential road process. Produced from Step 7.
  * Aforementioned `df_impl_prep.csv`, `df_impl_prep_full.csv`, and `segments_geoid.csv`.

## Output Directories

* Step 0: `src/detector/inference`, `data/damage_detect`
* Steps 1-5: `data/derived`
* Steps 6-8: `output`

## Approximate Execution Times

* Step 0: 10-15 sec/image on CPU
* Step 1: 1-1.5 hours
* Step 2: 30 min-1 hour
* Step 3: ~ 1 min
* Step 4: 5-10 min
* Step 5: < 1 min
* Step 6: < 1 min
* Step 7: ~ 1 min
* Step 8: < 1 min
