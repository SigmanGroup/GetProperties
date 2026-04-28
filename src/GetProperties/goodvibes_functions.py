#!/usr/bin/env python3
# coding: utf-8

'''
Functions specifically for the GoodVibes thermochemical extraction
'''

import sys
import logging
import itertools
import multiprocessing

from pathlib import Path

import pandas as pd
from goodvibes.GoodVibes import ATMOS, GAS_CONSTANT
from goodvibes.io import level_of_theory
from goodvibes.thermo import calc_bbe
from goodvibes.vib_scale_factors import scaling_data_dict, scaling_data_dict_mod

from .utils import FILE_COLUMN_NAME
from .utils import configure_logger

logger = logging.getLogger(__name__)

# Format the logging (you don't have to edit this)
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
    datefmt='%m/%d/%Y:%H:%M:%S',  # Correct way to format the date
    handlers=[logging.StreamHandler(sys.stdout)]
)

def _get_goodvibes_freq_scale_factor(file: Path):
    '''
    Replicate the GoodVibes 3.2 automatic vibrational scale-factor lookup.

    Parameters
    ----------
    filename: str
        Output file to inspect.

    Returns
    -------
    freq_scale_factor: float
        Vibrational scale factor used by GoodVibes.
    '''
    configure_logger(debug=False)

    # Detect the level of theory the same way GoodVibes does.
    level = level_of_theory(file=file).upper()

    # Search the built-in GoodVibes scale-factor tables.
    for data in (scaling_data_dict, scaling_data_dict_mod):
        if level in data:
            logger.info('Found %s in GoodVibes scaling data.\tzpe_fac: %f\tType: %s', level, data[level].zpe_fac, type(data[level].zpe_fac))

            # This must be returned as type float because
            # It was specified as float32 (f4) in goodvibes
            return float(data[level].zpe_fac)

    # Match the GoodVibes fallback when no match is found.
    logger.warning('Defaulting to zpe_fac = 1.0 for %s', level)
    return 1.0


def _get_goodvibes_thermo_data(logfile: Path | str,
                               temp: float = 298.15,
                               spc: str = 'link'):
    '''
    Helper function that mimics the old GoodVibes workflow and returns
    thermochemical data as a dict.

    Parameters
    ----------
    logfile: str
        Gaussian/ORCA output file path. A bare stem is also accepted.

    temp: float
        Temperature in Kelvin.

    spc: str
        Single-point correction mode. Use 'link' to match the old code.

    Returns
    -------
    thermo_data: dict
        Thermochemical data extracted from the GoodVibes calc_bbe object.
    '''
    try:
        # Match the GoodVibes gas-phase default concentration when -c is not supplied.
        conc = ATMOS / (GAS_CONSTANT * temp)

        # Match GoodVibes automatic vibrational scale-factor detection.
        freq_scale_factor = _get_goodvibes_freq_scale_factor(logfile)

        # Call the real GoodVibes 3.2 thermochemistry engine directly.
        bbe = calc_bbe(
            file=logfile,
            QS='grimme',
            QH=False,
            s_freq_cutoff=100.0,
            H_FREQ_CUTOFF=100.0,
            temperature=temp,
            conc=conc,
            freq_scale_factor=freq_scale_factor,
            solv='none',
            spc=spc,
            invert=False,
            d3_term=0.0,
            cosmo=None,
            ssymm=False,
            mm_freq_scale_factor=False,
            inertia='global',
            g4=False,
        )

        # Return the same values the old version
        thermo_data =  pd.Series({
            FILE_COLUMN_NAME: logfile.name,
            'E_spc (Hartree)': bbe.sp_energy,
            'ZPE(Hartree)': bbe.zpe,
            'H_spc(Hartree)': bbe.enthalpy,
            'T*S': bbe.entropy * temp,
            'T*qh_S': bbe.qh_entropy * temp,
            'G(T)_spc(Hartree)': bbe.gibbs_free_energy,
            'qh_G(T)_spc(Hartree)': bbe.qh_gibbs_free_energy,
            'T': temp
        })

    except Exception as e:
        logger.error('Could not get GoodVibes thermochemical data for %s because %s', logfile.name, e)
        thermo_data =  pd.Series({
            FILE_COLUMN_NAME: logfile.name,
            'E_spc (Hartree)': None,
            'ZPE(Hartree)': None,
            'H_spc(Hartree)': None,
            'T*S': None,
            'T*qh_S': None,
            'G(T)_spc(Hartree)': None,
            'qh_G(T)_spc(Hartree)': None,
            'T': temp
        })

    return pd.DataFrame(thermo_data).transpose()


def get_goodvibes_data(dataframe: pd.DataFrame,
                       data_dir: Path,
                       temp: float = 298.15,
                       procs: int = 1):
    '''
    Extracts the following properties

    - E_spc (Hartree)
    - ZPE(Hartree)
    - H_spc(Hartree)
    - T*S
    - T*qh_S
    - G(T)_spc(Hartree)
    - qh_G(T)_spc(Hartree)
    - T

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing `FILE_COLUMN_NAME` column

    data_dir: Path
        Directory where the files are located

    temp: float
        Temperature in Kelvin

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the `FILE_COLUMN_NAME` column and
        the resultant descriptors
    '''
    files = [Path(data_dir / x) for x in dataframe[FILE_COLUMN_NAME].to_list()]

    args = zip(files,
            itertools.repeat(temp),
            itertools.repeat('link'))

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_goodvibes_thermo_data, args)

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('GoodVibes function completed.')
    return dataframe


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
        datefmt='%m/%d/%Y:%H:%M:%S',  # Correct way to format the date
        handlers=[logging.StreamHandler(sys.stdout)]
    )
