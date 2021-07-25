"""
File containing high-level functions for MRB preprocessing.
"""

from .main_segment_loader import load_main_segments
from .traffic_loader import load_traffic
from .bus_loader import load_bus
from .segment_matcher.segment_matcher import match_segments
from .crack_loader import load_crack

import pickle


def prep_road_traffic_bus():
    with open("../data/derived/detroit_borders.pickle", "rb") as f:
        detroit_borders = pickle.load(f)
    
    print("--- RUNNING MAIN SEGMENT LOADER (step 1) ---\n")
    load_main_segments()
    print("--- RUNNING TRAFFIC SEGMENT LOADER (step 1) ---\n")
    load_traffic(detroit_borders)
    print("--- RUNNING BUS SEGMENT LOADER (step 1) ---\n")
    load_bus(detroit_borders)
    

def prep_combine_datasets():
    print("--- RUNNING SEGMENT MATCHER (step 2) ---\n")
    match_segments()


def prep_crack():
    print("--- RUNNING CRACK DATA LOADER (step 3) ---\n")
    load_crack()