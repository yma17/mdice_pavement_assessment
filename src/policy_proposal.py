"""
"Main" file to run policy proposal end-to-end pipeline.

Steps of the process (in order):
* Run inference on road images using crack detection model (0.)
* Preprocess city-wide datasets
    1. load city-wide road data, traffic data, bus data
    2. combine datasets from step 1
    3. city-wide crack data, combine with step 2 output
    4. load census data
    5. load public asset, match to census blocks
* Run decision process
    6. MRB
    7. RRB
* Format results to be visualized on UI tool (8.)

The program takes in the following optional command line parameter:
    -p, --pipeline: which steps to run, in order. Default: '678'
        * Check README.md for existing file requirements for each step.

Decision parameters for MRB and RRB are stored under config/.
"""

from preprocess.preprocess_mrb import prep_road_traffic_bus, prep_crack
from preprocess.preprocess_mrb import prep_combine_datasets
from preprocess.preprocess_rrb import prep_census, prep_public_assets
from decision.mrb_proc import mrb_proc
from decision.rrb_proc import rrb_proc
from format_viz import format_viz

import argparse
import json
import subprocess
import glob

def run_steps(args):
    # Check validity of input
    for step in args.pipeline:
        if ord(step) < 48 or ord(step) > 56:
            raise ValueError("Invalid pipeline step id.")

    # Run pipeline steps
    for step in args.pipeline:
        if step == '0':
            print("--- RUNNING MODEL INFERENCE (step 0) ---\n")

            # download pretrained model weights if not found
            if len(glob.glob(f'detector/yolov5/weights/IMSC/*.pt')) == 0:
                cmd = "cd detector/yolov5 && bash scripts/download_IMSC_grddc2020_weights.sh"
                subprocess.run(cmd, shell=True)

            # run damage detector
            cmd = f"""cd detector/yolov5 && python3 detect.py \\
                --source ../images \\
                --output ../inference \\
                --agnostic-nms \\
                --augment && python3 convert.py
            """.strip()
            subprocess.run(cmd, shell=True)
            cmd = "mv detector/yolov5/damage.csv ../data/damage_detect/damage.csv"
            subprocess.run(cmd, shell=True)
        elif step == '1':
            prep_road_traffic_bus()
        elif step == '2':
            prep_combine_datasets()
        elif step == '3':
            prep_crack()
        elif step == '4':
            prep_census()
        elif step == '5':
            prep_public_assets()
        elif step == '6':
            with open("../config/mrb_config.json") as f:
                config = json.load(f)
                mrb_proc(config)
        elif step == '7':
            with open("../config/rrb_config.json") as f:
                config = json.load(f)
                rrb_proc(config)
        else:  # step == '8'
            format_viz()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pipeline', type=str, default='678')
    args = parser.parse_args()
    run_steps(args)