"""
File containing high-level functions for RRB preprocessing.
"""

from .acs_read import acs_read
from .acs_prep import acs_prep
from .asset_loader import load_assets


def prep_census():
    print("--- READING CENSUS ACS DATA (step 4) ---\n")
    acs_read()
    print("--- PREPPING CENSUS ACS DATA (step 4) ---\n")
    acs_prep()


def prep_public_assets():
    print("--- PREPPING PUBLIC ASSET DATA (step 5) ---\n")
    load_assets()