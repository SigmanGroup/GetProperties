#!/usr/bin/env python3
# coding: utf-8

'''
Code for comparing old vs. new values
'''

import re
import sys
import math
import logging
import itertools
import multiprocessing

from pprint import pprint
from pathlib import Path

import pandas as pd
import numpy as np
from natsort import natsorted

logger = logging.getLogger()

'''
# Ignore (not in the old DataFrame)
SASA_volume(Å³)
SASA_sphericity
wall_time (h)
SASA_surface_area(Å²)
volume(Bohr_radius³/mol)

# Not in old
dihedral_Cb_N_Ca_Ca1 (°)
dihedral_Cb_N_Ca_Ca2 (°)
dihedral_Ca_N_Cb_Cb1 (°)
dihedral_Ca_N_Cb_Cb2 (°)
plane_angle_Ca-Ca1-Ca2_Cb-Cb1-Cb2 (°)

# Not in new
dihedral_Cb_N_Ca_Ca1(°)
dihedral_Cb_N_Ca_Ca2(°)
dihedral_Ca_N_Cb_Cb1(°)
dihedral_Ca_N_Cb_Cb2(°)
planeangle_Ca_Ca1_Ca2_&_Cb_Cb1_Cb2(°)
'''

def main():
    '''
    Main entrypoint
    '''
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
        datefmt='%m/%d/%Y:%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

    new = pd.read_csv('./test_2/results/individual_properties.csv')
    old = pd.read_excel('./test_2/results/Aniline_product_conformer_properties.xlsx', sheet_name='Sheet1')

    # Drop the unnamed column
    old.drop(columns=['Unnamed: 0'], inplace=True)

    # Set index
    old.set_index('log_name', inplace=True, drop=True)
    old.index = [f'{x}.log' for x in old.index]
    old.index.name = 'file'
    new.set_index('file', inplace=True, drop=True)

    # Sort the indices
    old = old.reindex(natsorted(old.index))
    new = new.reindex(natsorted(new.index))

    all_columns = [x for x in old.columns]
    all_columns.extend([x for x in new.columns])
    all_columns = list(set(all_columns))

    special_columns = []

    # Print out columns for comparing specific columns
    for column in all_columns:
        if column not in old:
            special_columns.append(column)
            logger.info('%s ', column)
    print()
    for column in all_columns:
        if column not in new:
            special_columns.append(column)
            logger.info('%s ', column)

    special_columns = list(set(special_columns))

    # Compare items columns that are in both DataFrame
    columns_to_compare = natsorted([x for x in new.columns if x not in special_columns])
    _new_direct_compare = new[columns_to_compare].copy(deep=True)
    _old_direct_compare = old[columns_to_compare].copy(deep=True)

    diff = _new_direct_compare - _old_direct_compare

    diff.to_csv('./diff.csv')

    abs_percent_diff = 100 * diff.abs().div(_old_direct_compare.abs().replace(0, np.nan))
    max_percent_diff_per_column = abs_percent_diff.max(axis=0).sort_values(ascending=False)

    # Iterate over the worst performers
    for k, v in max_percent_diff_per_column.items():

        # Skip columns whose worst difference is below threshold
        if v < 0.5:
            continue

        for idx in _new_direct_compare.index:
            new_value = _new_direct_compare.loc[idx, k]
            old_value = _old_direct_compare.loc[idx, k]
            _diff = new_value - old_value

            if pd.isna(old_value) or np.isclose(old_value, 0.0):
                _percent_diff = np.nan
            else:
                _percent_diff = abs(_diff) / abs(old_value) * 100

            if pd.notna(_percent_diff) and _percent_diff >= 0.5:
                print(f'{idx}\t{k}\tnew: {round(new_value, 5)}\told: {round(old_value, 5)}\tdiff: {round(_diff, 4)} ({round(_percent_diff, 2)} %)')

    special_comparisons = {
        'dihedral_Cb_N_Ca_Ca1 (°)': 'dihedral_Cb_N_Ca_Ca1(°)',
        'dihedral_Cb_N_Ca_Ca2 (°)': 'dihedral_Cb_N_Ca_Ca2(°)',
        'dihedral_Ca_N_Cb_Cb1 (°)': 'dihedral_Ca_N_Cb_Cb1(°)',
        'dihedral_Ca_N_Cb_Cb2 (°)': 'dihedral_Ca_N_Cb_Cb2(°)',
        'plane_angle_Ca-Ca1-Ca2_Cb-Cb1-Cb2 (°)': 'planeangle_Ca_Ca1_Ca2_&_Cb_Cb1_Cb2(°)',
    }

    for _new_prop, _old_prop in special_comparisons.items():

        # Copy the two Series and keep the index
        _new_series = new[_new_prop].copy()
        _old_series = old[_old_prop].copy()

        # Compute the difference by matching index labels
        diff = _new_series - _old_series

        print(f'{_new_prop} vs {_old_prop}\tTotal absolute difference: {np.abs(diff).sum():.2f}\tax abs: {np.abs(diff).max():.2f}')


if __name__ == "__main__":
    main()