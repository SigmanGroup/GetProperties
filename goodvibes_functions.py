#!/usr/bin/env python3
# coding: utf-8

'''
Isolated GoodVibes functions
'''

import itertools
import multiprocessing

from pathlib import Path
import pandas as pd

import goodvibes
import goodvibes.GoodVibes as gv
print(gv.GVOptions())
import goodvibes.thermo as thermo
import goodvibes.io as io

print(gv.GVOptions())

def _run_goodvibes(file: Path, options) -> pd.DataFrame:
    '''
    Runs GoodVibes on a single file and returns a pd.DataFrame
    containing the log_name and a series of thermochemical features
    from GoodVibes.
    '''
    # create a text file for all output (required)
    logger = io.Logger("Goodvibes", 'output', False)


    try:
        file_data = io.getoutData(str(file.absolute()), options)

        # Carry out the thermochemical analysis - auto-detect the vibrational scaling factor
        # Turns of default value of 1
        options.freq_scale_factor = False
        level_of_theory = [file_data.functional + '/' + file_data.basis_set]
        options.freq_scale_factor, options.mm_freq_scale_factor = gv.get_vib_scale_factor(level_of_theory, options, logger)
        bbe_val = thermo.calc_bbe(file_data, options)

        properties = ['sp_energy', 'zpe', 'enthalpy', 'entropy', 'qh_entropy', 'gibbs_free_energy', 'qh_gibbs_free_energy']
        vals = [getattr(bbe_val, k) for k in properties]

        row_i = pd.Series({
                'log_name': file.name,
                'E_spc (Hartree)': vals[0],
                'ZPE(Hartree)': vals[1],
                'H_spc(Hartree)': vals[2],
                'T*S': vals[3]*options.temperature,
                'T*qh_S': vals[4]*options.temperature,
                'G(T)_spc(Hartree)': vals[5],
                'qh_G(T)_spc(Hartree)': vals[6],
                'T': options.temperature})

    except Exception as e:
        print(f'An exception has occurred: {e}')
        print(f'[ERROR] Unable to acquire GoodVibes energies for {file.name}')
        row_i = pd.Series({
                            'log_name': file.name,
                            'E_spc (Hartree)': "no data",
                            'ZPE(Hartree)': "no data",
                            'H_spc(Hartree)': "no data",
                            'T*S': "no data",
                            'T*qh_S': "no data",
                            'G(T)_spc(Hartree)': "no data",
                            'qh_G(T)_spc(Hartree)': "no data",
                            'T': "no data"})

    return pd.DataFrame(row_i).transpose()

def get_goodvibes_e(dataframe: pd.DataFrame,
                    data_dir: Path,
                    temp: float = 298.15,
                    procs: int = 1) -> pd.DataFrame:
    '''
    Runs GoodVibes on a dataframe containing the log_name column. These
    files should be found in the data_dir.
    '''
    # Make a results dataframe
    e_dataframe = pd.DataFrame(columns=['log_name', 'E_spc (Hartree)', 'ZPE(Hartree)', 'H_spc(Hartree)', 'T*S', 'T*qh_S', 'G(T)_spc(Hartree)', 'qh_G(T)_spc(Hartree)', 'T'])

    # Set goodvibes options
    options = gv.GVOptions()
    options.spc = 'link'
    options.temperature = temp

    files = [Path(data_dir / x) for x in dataframe['log_name'].to_list()]

    # Multiprocessing goodvibes is difficult because it will pickle some
    # strange objects and try to import everything a million times.
    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_run_goodvibes, zip(files, itertools.repeat(options)))

    #results = []
    #for file in files:
    #    results.append(_run_goodvibes(file, options=options))

    e_dataframe = pd.concat(results)

    print('[INFO] Goodvibes has completed.')

    # Merge the old and new dataframes based on the log name
    e_dataframe.set_index('log_name', inplace=True, drop=True)
    dataframe.set_index('log_name', inplace=True, drop=True)

    dataframe = pd.concat([dataframe, e_dataframe], axis=1)
    dataframe.reset_index(inplace=True)
    return dataframe
