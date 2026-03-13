#!/usr/bin/env python3
# coding: utf-8

'''
New parallelized functions for get_properties
'''

import re
import math
import logging
import itertools
import multiprocessing

from pprint import pprint
from pathlib import Path

import pandas as pd
import numpy as np

import morfeus
from morfeus import Sterimol
from morfeus import BuriedVolume
from morfeus import Pyramidalization
from morfeus import SASA

from utils import get_filecont, get_outstreams, get_geom
from utils import get_specdata
from utils import FILE_COLUMN_NAME
from utils import configure_logger

logger = logging.getLogger()

# Misc
homo_pattern = re.compile(r"Alpha  occ. eigenvalues", re.DOTALL)
polarizability_pattern = re.compile(r"", re.DOTALL)
POLARIZABILITY_TABLE_PATTERN = re.compile(r'(?<=Dipole polarizability, Alpha \(input orientation\).)(.*?)(?=------------------|First dipole hyperpolarizability)', re.DOTALL)
dipole_pattern = "Dipole moment (field-independent basis, Debye)"
volume_pattern = re.compile("Molar volume =")
wall_time_pattern = re.compile(r'Elapsed time:\s+(\d+)\s+days\s+(\d+)\s+hours\s+(\d+)\s+minutes\s+(\d+(?:\.\d+)?)\s+seconds\.')

# NBO/NPA
nbo_os_pattern = re.compile("beta spin orbitals")
npa_pattern = re.compile("Summary of Natural Population Analysis:")

# NMR
nmrstart_pattern = " SCF GIAO Magnetic shielding tensor (ppm):\n"
nmrend_pattern = re.compile("End of Minotr F.D.")
nmrend_pattern_os = re.compile("g value of the free electron")

# ChelpG
chelpg1_pattern = re.compile("(CHELPG)")
chelpg2_pattern = re.compile("Charges from ESP fit")

# Hirsh
hirshfeld_pattern = re.compile("Hirshfeld charges, spin densities, dipoles, and CM5 charges")


def _get_frontier_orbs(file: Path) -> pd.DataFrame:
    '''
    Extracts the homo, lumo energies and derived values of last job in file
    of a Gaussian16 logfile.

    Parameters
    ----------
    file: Path
        Gaussian16 logfile

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    try:
        # Read the contents of the log file
        filecont, error = get_filecont(str(file.absolute()), split=True)

        if error != "":
            logger.error('Error reading in %s\t%s', file.name, error)
            row_i = pd.Series({FILE_COLUMN_NAME: file.name,
                               'HOMO': "no data",
                               'LUMO': "no data",
                               "μ": "no data",
                               "η": "no data",
                               "ω": "no data"})
        else:
            frontierout = []
            index = 0
            for line in filecont[::-1]:
                if homo_pattern.search(line):
                    index += 1 #index ensures only the first entry is included
                    if index == 1:
                        homo = float(str.split(line)[-1])
                        lumo = float(str.split(filecont[filecont.index(line) + 1])[4])
                        mu = (homo+lumo) / 2 # chemical potential or negative of molecular electronegativity
                        eta = lumo-homo # hardness/softness
                        omega = round(mu**2/(2*eta),5) # electrophilicity index
                        frontierout.append(homo)
                        frontierout.append(lumo)
                        frontierout.append(mu)
                        frontierout.append(eta)
                        frontierout.append(omega)

            #this adds the data from the frontierout into the new property df
            row_i = pd.Series({FILE_COLUMN_NAME:file.name, 'HOMO': frontierout[0], 'LUMO': frontierout[1], "μ": frontierout[2], "η": frontierout[3], "ω": frontierout[4]})

    except Exception as e:
        logger.error('Exception in _get_frontier_orbs for %s\t%s', file.name, e)
        row_i = pd.Series({FILE_COLUMN_NAME:file.name, 'HOMO': "no data", 'LUMO': "no data", "μ": "no data", "η": "no data", "ω": "no data"})

    return pd.DataFrame(row_i).transpose()


def get_frontierorbs(dataframe: pd.DataFrame,
                     data_dir: Path,
                     procs: int = 1):
    '''
    Extracts the homo, lumo energies and derived values of last job in file
    from files in the <FILE_COLUMN_NAME> column of a DataFrame

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    files = [Path(data_dir / x) for x in dataframe[FILE_COLUMN_NAME].to_list()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_frontier_orbs, files)

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('Frontier orbitals function has completed.')
    return dataframe


def _get_polarizability(file: Path) -> pd.DataFrame:
    '''
    Extracts polarizability information from a G16 log file.
    The isotropic and anisotropic polarizability are extracted
    from the last table present in the file.

    Parameters
    ----------
    file: Path
        Gaussian16 logfile

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    try:
        # Don't split the log lines so we can use regex to isolate tables
        filecont, error = get_filecont(file, split=False)

        if error != "":
            logger.error('Error in _get_polarizability: %s\t%s', file.name, error)
            row_i = pd.Series({FILE_COLUMN_NAME:file.name, 'polar_iso(au)': "no data", 'polar_aniso(au)': "no data"})

        else:
            matches = re.findall(POLARIZABILITY_TABLE_PATTERN, filecont)

            if len(matches) == 0:
                logger.error('No polarizability information was found using POLARIZABILITY_TABLE_PATTERN')
            else:
                # Get the last match
                match = matches[-1].split('\n')

                # Alpha iso is on index 4 and Alpha aniso is on index 5
                alpha_iso = float(match[4].split()[1].replace('D', "E"))
                alpha_aniso = float(match[5].split()[1].replace("D", "E"))

                row_i = pd.Series({FILE_COLUMN_NAME: file.name, 'polar_iso(au)': alpha_iso, 'polar_aniso(au)': alpha_aniso})

    except Exception as e:
        logger.error('Unable to acquire polarizability for: %s because %s', file.name, e)
        row_i = pd.Series({FILE_COLUMN_NAME:file.name, 'polar_iso(Debye)': "no data", 'polar_aniso(Debye)': "no data"})

    return pd.DataFrame(row_i).transpose()


def get_polarizability(dataframe: pd.DataFrame,
                       data_dir: Path,
                       procs: int = 1) -> pd.DataFrame:
    '''
    Extracts the polarizability from files in the <FILE_COLUMN_NAME> column
    of a DataFrame

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''

    files = [Path(data_dir / x) for x in dataframe[FILE_COLUMN_NAME].to_list()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_polarizability, files)

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('Polarizability function has completed.')
    return dataframe


def _get_dipole(file: Path) -> pd.DataFrame:
    '''
    Extracts the dipole from a Gaussian16 logfile.

    Parameters
    ----------
    file: Path
        Gaussian16 logfile

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    try: #try to get the data
        filecont, error = get_filecont(file, split=True) #read the contents of the log file
        if error != "":
            logger.error('Error in _get_polarizability: %s\t%s', file.name, error)
            row_i = pd.Series({FILE_COLUMN_NAME:file.name, 'dipole(Debye)': "no data"})
        else:
            dipole = []
            for i in range(len(filecont) - 1, 0, -1): #search filecont in backwards direction
                if dipole_pattern in filecont[i]:
                    dipole.append(float(str.split(filecont[i + 1])[-1]))
            #this adds the data from the first dipole entry (corresponding to the last job in the file) into the new property df
            row_i = pd.Series({FILE_COLUMN_NAME:file.name, 'dipole(Debye)': dipole[0]})
    except Exception as e:
        logger.error('Unable to acquire dipole for: %s because %s', file.name, e)
        row_i = pd.Series({FILE_COLUMN_NAME:file.name, 'dipole(Debye)': "no data"})

    return pd.DataFrame(row_i).transpose()


def get_dipole(dataframe: pd.DataFrame,
               data_dir: Path,
               procs: int = 1) -> pd.DataFrame:
    '''
    Extracts the dipole from files in the <FILE_COLUMN_NAME> column
    of a DataFrame

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''

    files = [Path(data_dir / x) for x in dataframe[FILE_COLUMN_NAME].to_list()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_dipole, files)

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('Dipole function has completed.')
    return dataframe


def _get_volume(file: Path) -> pd.DataFrame:
    '''
    Extracts the molecular volume from a Gaussian16 logfile.

    Parameters
    ----------
    file: Path
        Gaussian16 logfile

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    try:
        filecont, error = get_filecont(file, split=True) #read the contents of the log file
        if error != "":
            logger.error('Error in _get_volume: %s\t%s', file.name, error)
            row_i = pd.Series({FILE_COLUMN_NAME: file.name, 'volume(Bohr_radius³/mol)': "no data"})

        else:
            volume = []
            for line in filecont:
                if volume_pattern.search(line):
                    volume.append(line.split()[3])
            #this adds the data into the new property df
            row_i = pd.Series({FILE_COLUMN_NAME:file.name, 'volume(Bohr_radius³/mol)': float(volume[0])})

    except Exception as e:
        logger.error('Unable to acquire volume for: %s because %s', file.name, e)
        row_i = pd.Series({FILE_COLUMN_NAME: file.name, 'volume(Bohr_radius³/mol)': "no data"})

    return pd.DataFrame(row_i).transpose()


def get_volume(dataframe: pd.DataFrame,
               data_dir: Path,
               procs: int = 1) -> pd.DataFrame:
    '''
    Extracts the volume from files in the <FILE_COLUMN_NAME> column
    of a DataFrame

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    files = [Path(data_dir / x) for x in dataframe[FILE_COLUMN_NAME].to_list()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_volume, files)

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('Volume function has completed.')
    return dataframe


def _get_SASA(file: Path) -> pd.DataFrame:
    '''
    Extracts the SASA from a Gaussian16 logfile.

    Parameters
    ----------
    file: Path
        Gaussian16 logfile

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    try:

        if file.suffix == '.xyz':
            elements, coordinates = morfeus.read_xyz(file)
        else:
            streams, error = get_outstreams(file)
            if error != "":
                logger.error('Error in _get_sasa: %s\t%s', file.name, error)
                row_i = pd.Series({FILE_COLUMN_NAME:file.name,
                                'SASA_surface_area(Å²)': "no data",
                                'SASA_volume(Å³)': "no data",
                                'SASA_sphericity': "no data"})
            else:
                log_coordinates = get_geom(streams)
                elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
                coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])

            sasa = SASA(elements, coordinates) #calls morfeus

            sphericity = np.cbrt((36*math.pi*sasa.volume**2))/sasa.area

            row_i = pd.Series({FILE_COLUMN_NAME:file.name,
                               'SASA_surface_area(Å²)': sasa.area,
                               'SASA_volume(Å³)': sasa.volume, #volume inside the solvent accessible surface area
                               'SASA_sphericity': sphericity})
    except Exception as e:
        logger.error('Unable to acquire SASA for: %s because %s', file.name, e)
        row_i = pd.Series({FILE_COLUMN_NAME:file.name,
                            'SASA_surface_area(Å²)': "no data",
                            'SASA_volume(Å³)': "no data",
                            'SASA_sphericity': "no data"})

    return pd.DataFrame(row_i).transpose()


def get_SASA(dataframe: pd.DataFrame,
             data_dir: Path,
             procs: int = 1) -> pd.DataFrame:
    '''
    Uses MORFEUS to calculate solvent accessible surface area (SASA) for files
    in the <FILE_COLUMN_NAME> column of a DataFrame. If you want to SASA with different
    probe radii, MORFEUS has this functionality, but it has not been implemented here.

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    files = [Path(data_dir / x) for x in dataframe[FILE_COLUMN_NAME].to_list()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_SASA, files)

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('SASA function has completed.')
    return dataframe


def _get_nbo(row: pd.Series) -> pd.DataFrame:
    '''
    The input is a pd.Series (row) that containes a <FILE_COLUMN_NAME>
    item that is a string representation Path to the .log file.

    The remaining items should be atom_label: atom_number (1-indexed)
    pairs that indicate the atom for which the NBO values will
    be extracted

    Parameters
    ----------
    file: Path
        Gaussian16 logfile

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    assert FILE_COLUMN_NAME in row.keys()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])

    # Get a results dictionary that contains all the keys of the atoms we're looking at
    results = {f'NBO_charge_{k}': 'no data' for k in [x for x in row.keys() if FILE_COLUMN_NAME not in x]}

    file = Path(row[FILE_COLUMN_NAME])

    try:
        # Read the contents of the log file
        filecont, error = get_filecont(file, split=True)
        if error != "":
            for k in results.keys():
                results[k] = 'no data'
        else:
            nbo, nbostart, nboout, skip = [], 0, "", 0

            #this section finds the line (nbostart) where the nbo data is located
            for i in range(len(filecont)-1,0,-1): #search the file contents for the phrase "beta spin orbitals" to check for open shell molecules
                if re.search(nbo_os_pattern,filecont[i]) and skip == 0:
                    skip = 2 # retrieve only combined orbitals NPA in open shell molecules

                # Search the file content for the phrase which indicates the start of the NBO section
                if npa_pattern.search(filecont[i]):
                    if skip != 0:
                        skip = skip - 1
                        continue
                    # Skips the set number of lines between the search key and the start of the table
                    nbostart = i + 6
                    break

            if nbostart == 0:
                logger.error('Error in _get_nbo. No NPA found in %s.', file.name)
                for k in results.keys():
                    results[k] = 'no data'
            else:

                #this section splits the table where nbo data is located into just
                # the atom number and charge to generate a list of lists (nbo)
                ls = []
                for line in filecont[nbostart:]:
                    if "==" in line: break
                    ls = [str.split(line)[1],str.split(line)[2]]
                    nbo.append(ls)

                #this uses the nbo list to return only the charges for only the atoms of interest as a list (nboout)
                for atom_label in [x for x in row.keys() if FILE_COLUMN_NAME not in x]:
                    specific_atom_number_we_are_looking_at = row[atom_label]
                    data_for_this_atom = get_specdata([str(specific_atom_number_we_are_looking_at)], nbo)[0]
                    results[f'NBO_charge_{atom_label}'] = data_for_this_atom

    except Exception as e:
        logger.error('Unable to acquire NBO for: %s because %s', file.name, e)

    results[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(results)).transpose()


def get_nbo(dataframe: pd.DataFrame,
            atom_list: list[str],
            data_dir: Path,
            procs: int = 1) -> pd.DataFrame:
    '''
    Gets the nbo for all atoms (a_list, form ["C1", "C4", "O2"]) in a
    dataframe that contains file name and atom number (1-indexed)

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    interesting_columns = [FILE_COLUMN_NAME]
    interesting_columns.extend(atom_list)
    calculation_df = dataframe[interesting_columns].copy(deep=True)

    # Convert the <FILE_COLUMN_NAME> column to path
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]

    # Get the rows of the dataframe that we will use as input for parallelization
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_nbo, calculation_rows)

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('NBO function has completed.')
    return dataframe


def _get_nmr(row: pd.Series) -> pd.DataFrame:
    '''
    The input is a pd.Series (row) that containes a <FILE_COLUMN_NAME>
    item that is a string representation Path to the .log file.

    Extracts the scalar isotropic shielding for the atoms specified
    in a_list. This value is inversely related to the chemical shift
    of the nucleus and is orientation independent.

    Parameters
    ----------
    file: Path
        Gaussian16 logfile

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    assert FILE_COLUMN_NAME in row.keys()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])

    # Get a results dictionary that contains all the keys of the atoms we're looking at
    results = {f'NMR_isotropic_shielding_{k}': None for k in [x for x in row.keys() if FILE_COLUMN_NAME not in x]}

    file = Path(row[FILE_COLUMN_NAME])

    try:
        filecont, error = get_filecont(file, split=True) #read the contents of the log file
        if error != "":
            for k in results.keys():
                results[k] = None
        else:
            if nmrstart_pattern in filecont:
                start = filecont.index(nmrstart_pattern)+1
                for i in range(start,len(filecont),1):
                    if nmrend_pattern.search(filecont[i]) or nmrend_pattern_os.search(filecont[i]):
                        end = i
                        break

            if start == 0:
                logger.error('Error in _get_nmr. No NMR found in %s.', file.name)
                for k in results.keys():
                    results[k] = None
            else:
                atoms = int((end - start) / 5) #total number of atoms in molecule (there are 5 lines generated per atom)
                nmr = []
                for atom in range(atoms):
                    element = str.split(filecont[start+5*atom])[1]
                    shift_s = str.split(filecont[start+5*atom])[4]
                    nmr.append([element,shift_s])

                for atom_label in [x for x in row.keys() if FILE_COLUMN_NAME not in x]:
                    specific_atom_number_we_are_looking_at = row[atom_label]
                    data_for_this_atom = get_specdata([str(specific_atom_number_we_are_looking_at)], nmr)[0]
                    results[f'NMR_isotropic_shielding_{atom_label}'] = data_for_this_atom

    except Exception as e:
        logger.error('Unable to acquire NMR for: %s because %s', file.name, e)

    results[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(results)).transpose()


def get_nmr(dataframe: pd.DataFrame,
            atom_list: list[str],
            data_dir: Path,
            procs: int = 1) -> pd.DataFrame:
    '''
    Gets the NMR for all atoms (a_list, form ["C1", "C4", "O2"]) in a
    dataframe that contains file name and atom number (1-indexed)

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    atom_list: list[str]
        List of atom identifiers for which the property is collected

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    interesting_columns = [FILE_COLUMN_NAME]
    interesting_columns.extend(atom_list)
    calculation_df = dataframe[interesting_columns].copy(deep=True)

    # Convert the <FILE_COLUMN_NAME> column to path
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]

    # Get the rows of the dataframe that we will use as input for parallelization
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_nmr, calculation_rows)

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('NMR function has completed.')
    return dataframe


def _get_distance(row: pd.Series, dist_list: list[list]) -> pd.Series:
    '''
    The input is a pd.Series (row) that containes a <FILE_COLUMN_NAME>
    item that is a string representation Path to the .log file.

    Parameters
    ----------
    row: pd.Series
        Gaussian16 logfile

    dist_list: list[list]
        A list of lists of ints where the ints are the 1-index atom indices.

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    assert FILE_COLUMN_NAME in row.keys()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])

    # Get a results dictionary that contains all the keys of the atoms we're looking at
    #'distance_' + str(disttitle_list[a]) + '
    atom_number_name_dict = {int(number):label for label, number in row.items() if label != FILE_COLUMN_NAME}
    #atom_name_number_dict = {label:int(number) for label, number in row.items() if label != FILE_COLUMN_NAME}

    results = {f'distance_{k[0]}_{k[1]}(Å)': 'no data' for k in dist_list}

    file = Path(row[FILE_COLUMN_NAME])

    try:
        filecont, error = get_filecont(file, split=True) # Read the contents of the log file
        if error != "":
            for k in results.keys():
                results[k] = 'no data'
        else:
            streams, error = get_outstreams(file)

            if error != "":
                logger.error('Error in _get_distance. %s.', error)
            else:

                geom = get_geom(streams)

                # Iterate over all of the atom name
                for dist in dist_list:

                    # Convert the atom names to atom indices (0-indexed)
                    atom_a = int(row[dist[0]] - 1)
                    atom_b = int(row[dist[1]] - 1)
                    a = geom[atom_a][:4]
                    b = geom[atom_b][:4]
                    ba = np.array(a[1:]) - np.array(b[1:])
                    distance_value = round(np.linalg.norm(ba), 5)
                    results[f'distance_{atom_number_name_dict[atom_a + 1]}_{atom_number_name_dict[atom_b + 1]}(Å)'] = distance_value

    except Exception as e:
        logger.error('Unable to acquire distance for: %s because %s', file.name, e)

    results[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(results)).transpose()


def get_distance(dataframe: pd.DataFrame,
                 dist_list: list[list[str]],
                 data_dir: Path,
                 procs: int = 1) -> pd.DataFrame:
    '''
    Gets the distance between atoms defined by a list of list of strings
    where the innermost strings are the atom labels in the dataframe
    (dist_list, form [[C1, O2], [C4, C1]]) in a dataframe that contains
    <FILE_COLUMN_NAME> and atom number.

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    dist_list: list[list[str]]
        List of lists that contain the pairs of atom labels for which
        the distance will be calculated.

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''

    interesting_columns = [FILE_COLUMN_NAME]
    interesting_columns.extend(list(set([x for xs in dist_list for x in xs])))
    calculation_df = dataframe[interesting_columns].copy(deep=True)

    # Convert the <FILE_COLUMN_NAME> column to path
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]

    # Get the rows of the dataframe that we will use as input for parallelization
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_distance, zip(calculation_rows, itertools.repeat(dist_list)))

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('Distance function has completed.')
    return dataframe


def _get_angles(row: pd.Series, angle_list: list[list]) -> pd.Series:
    '''
    The input is a pd.Series (row) that containes a <FILE_COLUMN_NAME>
    item that is a string representation Path to the .log file.

    The remaining items should be atom_label: atom_number (1-indexed)
    pairs that indicate the atom for which the angle values will
    be extracted

    Parameters
    ----------
    row: pd.Series
        Series that contains <FILE_COLUMN_NAME> which is a path to the
        file to be used for angle calculation as well as the
        atom labels

    angle_list: list[list[str]]
        List of list of strings where the strings are the atom labels
        that make up the angle of interest.

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    assert FILE_COLUMN_NAME in row.keys()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])

    # Get a results dictionary that contains all the keys of the atoms we're looking at
    atom_number_name_dict = {int(number): label for label, number in row.items() if label != FILE_COLUMN_NAME}

    results = {f'angle_{k[0]}_{k[1]}_{k[2]}(°)':'no data' for k in angle_list}

    file = Path(row[FILE_COLUMN_NAME])

    try:
        filecont, error = get_filecont(file, split=True) #read the contents of the log file
        if error != "":
            for k in results.keys():
                results[k] = 'no data'
        else:
            streams, error = get_outstreams(file)

            if error != "":
                logger.error('Error in _get_angles. %s.', error)
            else:
                # Get the geometry
                geom = get_geom(streams)

                # Iterate over all of the atom name
                for angle in angle_list:

                    # Get the atom coordinates
                    atom_a = geom[int(row[angle[0]]) - 1][:4]
                    atom_b = geom[int(row[angle[1]]) - 1][:4]
                    atom_c = geom[int(row[angle[2]]) - 1][:4]

                    ba = np.array(atom_a[1:]) - np.array(atom_b[1:])
                    bc = np.array(atom_c[1:]) - np.array(atom_b[1:])
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

                    # This returns radians, but we convert to degrees with np.degrees
                    anglevalue = np.degrees(np.arccos(cosine_angle))

                    results[f'angle_{angle[0]}_{angle[1]}_{angle[2]}(°)'] = anglevalue

    except Exception as e:
        logger.error('Unable to acquire angle for: %s because %s', file.name, e)

    results[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(results)).transpose()


def get_angles(dataframe: pd.DataFrame,
               angle_list: list[list[str]],
               data_dir: Path,
               procs: int = 1) -> pd.DataFrame:
    '''
    Gets the angle between three atoms defined by a list of list of strings
    where the innermost strings are the atom labels in the dataframe
    (angle_list, form [[O3, C1, O2], [C4, C1, O3]]) in a dataframe that contains
    <FILE_COLUMN_NAME> and atom number.

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    angle_list: list[list[str]]
        List of lists that contain the pairs of atom labels for which
        the angle will be calculated.

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    interesting_columns = [FILE_COLUMN_NAME]
    interesting_columns.extend(list(set([x for xs in angle_list for x in xs])))
    calculation_df = dataframe[interesting_columns].copy(deep=True)

    # Convert the <FILE_COLUMN_NAME> column to path
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]

    # Get the rows of the dataframe that we will use as input for parallelization
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_angles, zip(calculation_rows, itertools.repeat(angle_list)))

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('Angle function has completed.')
    return dataframe


def _get_dihedral(row: pd.Series, dihedral_list: list[list]) -> pd.Series:
    '''
    The input is a pd.Series (row) that containes a <FILE_COLUMN_NAME>
    item that is a string representation Path to the .log file.

    The remaining items should be atom_label: atom_number (1-indexed)
    pairs that indicate the atom for which the angle values will
    be extracted

    Parameters
    ----------
    row: pd.Series
        Series that contains <FILE_COLUMN_NAME> which is a path to the
        file to be used for dihedral calculation as well as the
        atom labels

    angle_list: list[list[str]]
        List of list of strings where the strings are the atom labels
        that make up the angle of interest.

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    assert FILE_COLUMN_NAME in row.keys()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])

    # Get a results dictionary that contains all the keys of the atoms we're looking at
    atom_number_name_dict = {int(number):label for label, number in row.items() if label != FILE_COLUMN_NAME}

    results = {f'dihedral_{k[0]}_{k[1]}_{k[2]}_{k[3]} (°)':'no data' for k in dihedral_list}

    file = Path(row[FILE_COLUMN_NAME])

    try:
        filecont, error = get_filecont(file, split=True) #read the contents of the log file
        if error != "":
            for k in results.keys():
                results[k] = 'no data'
        else:
            streams, error = get_outstreams(file)

            if error != "":
                logger.error('Error in _get_dihedrals. %s.', error)
            else:

                geom = get_geom(streams)

                # Iterate over all of the atom name
                for dihedral in dihedral_list:

                    if len(dihedral) != 4:
                        raise ValueError(f'Dihedral angles must be specified by a list of 4 labels not {dihedral}')
                    a = geom[int(row[dihedral[0]]) - 1][:4] # atom coords
                    b = geom[int(row[dihedral[1]]) - 1][:4]
                    c = geom[int(row[dihedral[2]]) - 1][:4]
                    d = geom[int(row[dihedral[3]]) - 1][:4]

                    ab = np.array([a[1]-b[1],a[2]-b[2],a[3]-b[3]]) # vectors
                    bc = np.array([b[1]-c[1],b[2]-c[2],b[3]-c[3]])
                    cd = np.array([c[1]-d[1],c[2]-d[2],c[3]-d[3]])

                    n1 = np.cross(ab,bc) # normal vectors
                    n2 = np.cross(bc,cd)

                    dihedral_value = round(np.degrees(np.arccos(np.dot(n1,n2) / (np.linalg.norm(n1)*np.linalg.norm(n2)))),3)
                    results[f'dihedral_{dihedral[0]}_{dihedral[1]}_{dihedral[2]}_{dihedral[3]} (°)'] = dihedral_value

    except Exception as e:
        logger.error('Unable to acquire dihedral angle for: %s because %s', file.name, e)

    results[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(results)).transpose()


def get_dihedral(dataframe: pd.DataFrame,
                 dihedral_list: list[list[str]],
                 data_dir: Path,
                 procs: int = 1) -> pd.DataFrame:
    '''
    Gets the dihedral angle of four atoms defined by a list of list of strings
    where the innermost strings are the atom labels in the dataframe
    (dihedral_list, form [['O1', 'C1', 'CA', 'C2']]) in a dataframe that contains
    <FILE_COLUMN_NAME> and atom number.

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    dihedral_list: list[list[str]]
        List of list of strings where the strings are the atom labels
        that make up the angle of interest.

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    interesting_columns = [FILE_COLUMN_NAME]
    interesting_columns.extend(list(set([x for xs in dihedral_list for x in xs])))
    calculation_df = dataframe[interesting_columns].copy(deep=True)

    # Convert the <FILE_COLUMN_NAME> column to path
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]

    # Get the rows of the dataframe that we will use as input for parallelization
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_dihedral, zip(calculation_rows, itertools.repeat(dihedral_list)))

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('Dihedral function has completed.')
    return dataframe


def _get_vbur(row: pd.Series,
              radii: list[float],
              a_list: list[str]) -> pd.Series:
    '''
    Uses morfeus to calculate vbur at a single radius
    for an atom (a1) in df

    Parameters
    ----------
    row: pd.Series
        Series that contains <FILE_COLUMN_NAME> which is a path to the
        file to be used for vbur calculation as well as the
        atom labels

    radii: list[float]
        List of radii at which vbur is calculated

    a_list: list[str]
        List of strings where the strings are the atom labels
        for the atoms for which vbur will be calculated

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    assert FILE_COLUMN_NAME in row.keys()

    # Get a results dictionary that contains all the keys of the atoms we're looking at
    atom_number_name_dict = {int(number): label for label, number in row.items() if label != FILE_COLUMN_NAME}

    # Make a results dictionary
    results = {}
    for atom in a_list:
        for radius in radii:
            results[f'%Vbur_{atom}_{str(radius)}Å'] = 'no_data'

    file = Path(row[FILE_COLUMN_NAME])

    try:
        filecont, error = get_filecont(file, split=True)  # read the contents of the log file
        if error != '':
            streams, error = get_outstreams(file)
            if error != '':
                logger.error('Error in _get_vbur. %s.', error)
            else:
                geom = get_geom(streams)

                for atom in a_list:
                    for radius in radii:
                        elements = np.array([geom[i][0] for i in range(len(geom))])

                        coordinates = np.array([np.array(geom[i][1:]) for i in range(len(geom))])

                        vbur = BuriedVolume(elements, coordinates, int(row[atom]), include_hs=True, radius=radius)

                        vbur_value = vbur.percent_buried_volume * 100
                        results[f'%Vbur_{atom}_{str(radius)}Å'] = vbur_value

    except Exception as e:
        logger.error('Unable to acquire vbur values for %s because %s', file.name, e)

    results[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(results)).transpose()


def get_vbur_scan(dataframe: pd.DataFrame,
                  data_dir: Path,
                  a_list: list[str],
                  start_r: float,
                  end_r: float,
                  step_size: float,
                  procs: int = 1) -> pd.DataFrame:
    '''
    Uses MORFEUS to scan vbur across a series of radii

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    data_dir: Path
        Directory where the files are located

    alist: list[str]
        List of strings where the strings are the atom labels

    start_r: float
        Initial radius of the scan for computing Vbur

    end_r: float
        End radius of the scan for computing Vbur

    step_size: float
        Step size for the scan

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors to use

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    assert FILE_COLUMN_NAME in dataframe.columns

    interesting_columns = [FILE_COLUMN_NAME]
    interesting_columns.extend(a_list)
    calculation_df = dataframe[interesting_columns].copy(deep=True)

    radii = [float(x) for x in np.arange(start_r, end_r, step_size)]

    if end_r not in radii:
        radii.append(end_r)

    # Convert the <FILE_COLUMN_NAME> column to path
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]

    # Get the rows of the dataframe that we will use as input for parallelization
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_vbur, zip(calculation_rows, itertools.repeat(radii), itertools.repeat(a_list)))

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('Vbur scan function has completed for radii: %s', str([round(r, 1) for r in radii]))
    return dataframe


def _get_sterimol_morfeus(row: pd.Series,
                          sterimol_list: list[str],
                          radius: float | None = None) -> pd.Series:
    '''
    Uses morfeus to calculate vbur at a single radius
    for an atom (a1) in df

    Parameters
    ----------
    row: pd.Series
        Series that contains <FILE_COLUMN_NAME> which is a path to the
        file to be used for vbur calculation as well as the
        atom labels

    radii: list[float]
        List of radii at which vbur is calculated

    a_list: list[str]
        List of strings where the strings are the atom labels
        for the atoms for which vbur will be calculated

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    assert FILE_COLUMN_NAME in row.keys()

    # Get a results dictionary that contains all the keys of the atoms we're looking at
    atom_number_name_dict = {int(number): label for label, number in row.items() if label != FILE_COLUMN_NAME}

    # This was present in the previous version for buried Sterimol
    # probably because of the addition of 0.5 Å in the calcualtion
    if radius is not None:
        radius -= 0.5

    # Make a results dictionary
    results = {}
    for atom_pair in sterimol_list:
        if radius is None:
            results[f'Sterimol_L_{atom_pair[0]}_{atom_pair[1]}(Å)_morfeus'] = 'no data'
            results[f'Sterimol_B1_{atom_pair[0]}_{atom_pair[1]}(Å)_morfeus'] = 'no data'
            results[f'Sterimol_B5_{atom_pair[0]}_{atom_pair[1]}(Å)_morfeus'] = 'no data'
        else:
            results[f'Buried_Sterimol_L_{atom_pair[0]}_{atom_pair[1]}_{radius}(Å)'] = 'no data'
            results[f'Buried_Sterimol_B1_{atom_pair[0]}_{atom_pair[1]}_{radius}(Å)'] = 'no data'
            results[f'Buried_Sterimol_B5_{atom_pair[0]}_{atom_pair[1]}_{radius}(Å)'] = 'no data'

    # Get the file we're looking at
    file = Path(row[FILE_COLUMN_NAME])

    try:
        filecont, error = get_filecont(file, split=True) #read the contents of the log file
        if error != "":
            pass
        else:
            streams, error = get_outstreams(file)

            if error != "":
                logger.error('Error in _get_sterimol_morfeus. %s.', error)
            else:
                geom = get_geom(streams)

                elements = np.array([geom[i][0] for i in range(len(geom))])
                coordinates = np.array([np.array(geom[i][1:]) for i in range(len(geom))])

                for sterimol_pair in sterimol_list:
                    if len(sterimol_pair) != 2:
                        raise ValueError(f'Number of atoms for Sterimol calculation must be 2 not {len(sterimol_pair)} ({sterimol_pair})')

                    atom_a = int(row[sterimol_pair[0]])
                    atom_b = int(row[sterimol_pair[1]])

                    if atom_a > len(geom):
                        raise ValueError(f'Atom A ({sterimol_pair[0]}, atom_number: {atom_a}) exceeds the length of the geometry for {file.name} ({len(geom)})')
                    if atom_a > len(geom):
                        raise ValueError(f'Atom B ({sterimol_pair[1]}, atom_number: {atom_b}) exceeds the length of the geometry for {file.name} ({len(geom)})')

                    # Get the Sterimol object
                    sterimol_calculation = Sterimol(elements, coordinates, atom_a, atom_b)

                    if radius is None:
                        results[f'Sterimol_L_{sterimol_pair[0]}_{sterimol_pair[1]}(Å)_morfeus'] = sterimol_calculation.L_value
                        results[f'Sterimol_B1_{sterimol_pair[0]}_{sterimol_pair[1]}(Å)_morfeus'] = sterimol_calculation.B_1_value
                        results[f'Sterimol_B5_{sterimol_pair[0]}_{sterimol_pair[1]}(Å)_morfeus'] = sterimol_calculation.B_5_value
                    else:
                        sterimol_calculation.bury(method="delete", sphere_radius=float(radius))
                        results[f'Buried_Sterimol_L_{sterimol_pair[0]}_{sterimol_pair[1]}_{radius}(Å)'] = sterimol_calculation.L_value
                        results[f'Buried_Sterimol_B1_{sterimol_pair[0]}_{sterimol_pair[1]}_{radius}(Å)'] = sterimol_calculation.B_1_value
                        results[f'Buried_Sterimol_B5_{sterimol_pair[0]}_{sterimol_pair[1]}_{radius}(Å)'] = sterimol_calculation.B_5_value
    except Exception as e:
        logger.error('Unable to acquire vbur values for: %s because %s', file.name, e)

    results[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(results)).transpose()


def get_sterimol_morfeus(dataframe: pd.DataFrame,
                         data_dir: Path,
                         sterimol_list: list[str],
                         radius: float | None = None,
                         procs: int = 1):
    '''
    Uses MORFEUS to get sterimol values

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    data_dir: Path
        Directory where the files are located

    sterimol_list: list[str]
        List of strings where the strings are the atom labels
        that make up the Sterimol axis of interest.

    radius: float | None
        If float, Sterimol is computed as Buried Sterimol

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    interesting_columns = [FILE_COLUMN_NAME]
    interesting_columns.extend(list(set([x for xs in sterimol_list for x in xs])))
    calculation_df = dataframe[interesting_columns].copy(deep=True)

    # Convert the <FILE_COLUMN_NAME> column to path
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]

    # Flatten the list of inputs to assess the label presence in the dataframe
    possible_columns = [x for xs in sterimol_list for x in xs]
    if not all([x in calculation_df.columns for x in possible_columns]):
        raise KeyError(f'Not all of the requested columns are in the dataframe. Requested: {possible_columns} Found: {list(dataframe.columns)}')

    # Get the rows of the dataframe that we will use as input for parallelization
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_sterimol_morfeus, zip(calculation_rows, itertools.repeat(sterimol_list), itertools.repeat(radius)))

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('Sterimol function has completed for %s. Radius: %s', sterimol_list, radius)
    return dataframe


def _get_chelpg(row: pd.Series, a_list: list[str]) -> pd.DataFrame:
    '''
    The input is a pd.Series (row) that containes a <FILE_COLUMN_NAME>
    item that is a string representation Path to the .log file.

    The remaining items should be atom_label: atom_number (1-indexed)
    pairs that indicate the atom for which the ChelpG values will
    be extracted

    Parameters
    ----------
    file: Path
        Gaussian16 logfile

    a_list: list[str]
        List of atom labels for which the ChelpG will be extracted

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    assert FILE_COLUMN_NAME in row.keys()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])

    # Get a results dictionary that contains all the keys of the atoms we're looking at
    results = {f'ChelpG_charge_{k}':'no data' for k in [x for x in row.keys() if FILE_COLUMN_NAME not in x]}

    file = Path(row[FILE_COLUMN_NAME])

    try:
        filecont, error = get_filecont(file, split=True)
        if error != "":
            for k in results.keys():
                results[k] = 'no data'
        else:
            chelpgstart, chelpg, error, chelpgout = 0, False, "", []

            # This section finds the line (chelpgstart) where the ChelpG data is located
            for i in range(len(filecont)-1,0,-1):
                if chelpg2_pattern.search(filecont[i]):
                    chelpgstart = i
                if chelpg1_pattern.search(filecont[i]):
                    chelpg = True
                    break

            if chelpgstart != 0 and chelpg == False:
                logger.error('No ChelpG charge scheme found in %s', file.name)
            elif chelpgstart == 0:
                logger.error('No ChelpG charge scheme found in %s', file.name)
            else:
                for atom_label in a_list:
                    results[f'ChelpG_charge_{atom_label}'] = filecont[chelpgstart+int(row[atom_label])+2].split()[-1]

    except Exception as e:
        logger.error('Unable to acquire ChelpG for: %s because %s', file.name, e)

    results[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(results)).transpose()


def get_chelpg(dataframe: pd.DataFrame,
               atom_list: list[str],
               data_dir: Path,
               procs: int = 1) -> pd.DataFrame:
    '''
    Gets the ChelpG ESP charge for all atoms (a_list, form ["C1", "C4", "O2"])
    in a dataframe that contains file name and atom number (1-indexed)

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    atom_list: list[str]
        List of atom identifiers for which the property is collected

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    interesting_columns = [FILE_COLUMN_NAME]
    interesting_columns.extend(atom_list)
    calculation_df = dataframe[interesting_columns].copy(deep=True)

    # Convert the <FILE_COLUMN_NAME> column to path
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]

    # Get the rows of the dataframe that we will use as input for parallelization
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_chelpg, zip(calculation_rows, itertools.repeat(atom_list)))

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('ChelpG function has completed')
    return dataframe


def _get_hirshfeld(row: pd.Series, a_list: list[str]) -> pd.DataFrame:
    '''
    The input is a pd.Series (row) that containes a <FILE_COLUMN_NAME>
    item that is a string representation Path to the .log file.

    The remaining items should be atom_label: atom_number (1-indexed)
    pairs that indicate the atom for which the Hirshfeld values will
    be extracted

    Parameters
    ----------
    file: Path
        Gaussian16 logfile

    a_list: list[str]
        List of atom labels for which the Hirshfeld will be extracted

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    assert FILE_COLUMN_NAME in row.keys()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])

    # Get a results dictionary that contains all the keys of the atoms we're looking at
    results = {}
    for atom_label in [x for x in row.keys() if FILE_COLUMN_NAME not in x]:
        results[f'Hirsh_charge_{atom_label}'] = 'no data'
        results[f'Hirsh_CM5_charge_{atom_label}'] = 'no data'
        results[f'Hirsh_atom_dipole_{atom_label}'] = 'no data'

    file = Path(row[FILE_COLUMN_NAME])

    try:
        filecont, error = get_filecont(file, split=True) #read the contents of the log file
        if error != "":
            for k in results.keys():
                results[k] = 'no data'
        else:
            hirshstart, error, hirshout = 0, '', []

            # This section finds the line (chelpgstart) where the ChelpG data is located
            for i in range(len(filecont)-1,0,-1):
                if hirshfeld_pattern.search(filecont[i]):
                    hirshstart = i
                    break

            if hirshstart == 0:
                logger.error('No Hirshfeld population analysis found in %s', file.name)
            else:
                for atom_label in a_list:
                    cont = filecont[hirshstart+int(row[atom_label]) + 1].split()
                    qh = cont[2] #using 0-indexing, this gets the value for Hirshfeld charge from the 2nd column
                    qcm5 = cont[7] #using 0-indexing, this gets the value for CM5 charge from the 7th column
                    d = np.linalg.norm(np.array((cont[4:8])))
                    results[f'Hirsh_charge_{atom_label}'] = str(qh)
                    results[f'Hirsh_CM5_charge_{atom_label}'] = str(qcm5)
                    results[f'Hirsh_atom_dipole_{atom_label}'] = str(d)

    except Exception as e:
        logger.error('Unable to acquire Hirshfeld for: %s because %s', file.name, e)

    results[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(results)).transpose()


def get_hirshfeld(dataframe: pd.DataFrame,
                  atom_list: list[str],
                  data_dir: Path,
                  procs: int = 1) -> pd.DataFrame:
    '''
    Gets the Hirshfeld charge, CM5, and diple for all atoms (a_list, form ["C1", "C4", "O2"])
    in a dataframe that contains file name and atom number (1-indexed)

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    atom_list: list[str]
        List of atom identifiers for which the property is collected

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    interesting_columns = [FILE_COLUMN_NAME]
    interesting_columns.extend(atom_list)
    calculation_df = dataframe[interesting_columns].copy(deep=True)

    # Convert the <FILE_COLUMN_NAME> column to path
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]

    # Get the rows of the dataframe that we will use as input for parallelization
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_hirshfeld, zip(calculation_rows, itertools.repeat(atom_list)))

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('Hirshfeld function has completed')
    return dataframe


def _get_pyramidalization(row: pd.Series,
                          a_list: list[str]) -> pd.DataFrame:
    '''
    The input is a pd.Series (row) that containes a <FILE_COLUMN_NAME>
    item that is a string representation Path to the .log file.

    The remaining items should be atom_label: atom_number (1-indexed)
    pairs that indicate the atom for which the Pyramidalization values will
    be computed with MORFEUS. The 3 closest atoms to the indicated atom
    will be used.

    Parameters
    ----------
    file: Path
        Gaussian16 logfile

    a_list: list[str]
        List of atom labels for which the Hirshfeld will be extracted

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    assert FILE_COLUMN_NAME in row.keys()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])

    # Get a results dictionary that contains all the keys of the atoms we're looking at
    results = {}
    for atom_label in [x for x in row.keys() if FILE_COLUMN_NAME not in x]:
        results[f'pyramidalization_Gavrish_{atom_label}_(°)'] = 'no data'
        results[f'pyramidalization_Agranat-Radhakrishnan_{atom_label}'] = 'no data'

    file = Path(row[FILE_COLUMN_NAME])

    try:
        # Read the contents of the log file
        filecont, error = get_filecont(file, split=True)

        if error != "":
            for k in results.keys():
                results[k] = 'no data'

        else:
            streams, error = get_outstreams(file)
            log_coordinates = get_geom(streams)
            elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
            coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])

            for atom_label in a_list:

                # Make Pyr object
                pyr = Pyramidalization(coordinates, int(row[atom_label]))

                results[f'pyramidalization_Gavrish_{atom_label}_(°)'] = pyr.P_angle
                results[f'pyramidalization_Agranat-Radhakrishnan_{atom_label}'] = pyr.P

    except Exception as e:
        logger.error('Unable to acquire pyramidalization for: %s because %s', file.name, e)

    results[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(results)).transpose()


def get_pyramidalization(dataframe: pd.DataFrame,
                         atom_list: list[str],
                         data_dir: Path,
                         procs: int = 1):

    #uses morfeus to calculate pyramidalization (based on the 3 atoms in closest proximity to the defined atom) for for all atoms (a_list, of form ["C1", "C4", "O2"]) in a dataframe that contains file name and atom number
    pyr_dataframe = pd.DataFrame(columns=[])

    interesting_columns = [FILE_COLUMN_NAME]
    interesting_columns.extend(atom_list)
    calculation_df = dataframe[interesting_columns].copy(deep=True)

    # Convert the <FILE_COLUMN_NAME> column to path
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]

    # Get the rows of the dataframe that we will use as input for parallelization
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_pyramidalization, zip(calculation_rows, itertools.repeat(atom_list)))

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('Pyramidalization function has completed')
    return dataframe


def _get_float_property_from_text(text: str,
                                  pattern: re.Pattern,
                                  return_type: str = 'all') -> list[float] | list[list[float]]:
    '''
    Extract one or more float groups from text using a regex pattern.

    Parameters
    ----------
    text: str
        Text to search.

    pattern: re.Pattern
        Compiled regex pattern containing one or more capture groups.

    return_type: str
        Options are "first", "last", and "all". If multiple instances of
        the pattern that should return multiple floats for each instance
        the pattern appears, this option will either return the first
        list of floats found, the last list of floats, or all of them.

    Returns
    -------
    values: list[float] | list[list[float]]
        Extracted float values. Returns a list of floats for "first" or
        "last", and a list of lists of floats for "all".
    '''

    # Find all matches
    matches = re.findall(pattern=pattern, string=text)

    # Handle no matches
    if not matches:
        raise ValueError('No matches found in text.')

    # Normalize one-group matches to tuple form
    # If there is only one extracted item, the first item in matches is a string
    if isinstance(matches[0], str):
        values = [[float(x)] for x in matches]
    else:
        values = [[float(z) for z in match] for match in matches]

    # Return all matches
    if return_type == 'all':
        return values
    elif return_type == 'first':
        return values[0]
    elif return_type == 'last':
        return values[-1]
    else:
        raise ValueError('return_type must be either "first", "last", or "all"')


def _get_wall_time(file: Path) -> pd.DataFrame:
    '''
    Extracts total wall time in hours for a Gaussian 16 logfile.

    Parameters
    ----------
    file: Path
        Gaussian16 logfile

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''
    configure_logger(debug=False)
    try:
        filecont, error = get_filecont(file, split=False)

        if error != '':
            logger.error('Error in _get_wall_time: %s\t%s', file.name, error)
            result_dict = {FILE_COLUMN_NAME: file.name, 'wall_time (h)': None}

        else:

            # Get the multiple floats for the walltime pattern
            all_times = _get_float_property_from_text(text=filecont,
                                                      pattern=wall_time_pattern,
                                                      return_type='all')

            days = np.sum([x[0] for x in all_times])
            hours = np.sum([x[1] for x in all_times])
            minutes = np.sum([x[2] for x in all_times])
            seconds = np.sum([x[3] for x in all_times])

            # Compute total hours
            total = hours + (days * 24) + (minutes / 60) + (seconds / 3600)

            result_dict = {FILE_COLUMN_NAME: file.name, 'wall_time (h)': total}

    except Exception as e:
        logger.error('Unable to acquire wall_time for: %s because %s', file.name, e)
        result_dict = {FILE_COLUMN_NAME:file.name, 'wall_time (h)': None}

    return pd.DataFrame(pd.Series(result_dict)).transpose()


def get_wall_time(dataframe: pd.DataFrame,
                  data_dir: Path,
                  procs: int = 1):
    '''
    Gets the total elapsed time a Gaussian 16 logfile took to complete
    in seconds.
    '''
    # Get the files from the DataFrame
    files = [Path(data_dir / x) for x in dataframe[FILE_COLUMN_NAME].to_list()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_wall_time, files)

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('Wall time function has completed.')
    return dataframe


def _get_plane_angle(row: pd.Series,
                     plane_a_names: list[str, str, str],
                     plane_b_names: list[str, str, str]) -> pd.Series:
    '''
    Parameters
    ----------
    row: pd.Series
        Gaussian16 logfile

    plane_a_indices: list[int, int, int]
        List of 3 atom names that make up the first plane.

    plane_b_indices: list[int, int, int]
        List of 3 atom names that make up the second plane.

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors.
    '''
    configure_logger(debug=False)

    # Get the file (which should be a Path at this point)
    file = Path(row[FILE_COLUMN_NAME])

    # Convert the names to indices for the geometry
    plane_a_indices = [row[x] for x in plane_a_names]
    plane_b_indices = [row[x] for x in plane_b_names]

    filecont, error = get_filecont(log=file, split=False)

    if not error:
        streams, error = get_outstreams(log=file)

    # Set up the results dictionary
    results = {f'plane_angle_{"-".join(plane_a_names)}_{"-".join(plane_b_names)} (º)': None}

    #try:
    # Get the geometry
    geom = get_geom(streams=streams)

    # Get the planes as 0-index
    plane_a_indices = [x - 1 for x in plane_a_indices]
    plane_b_indices = [x - 1 for x in plane_b_indices]

    # Get the xyz coordinates of the planes
    # We use everything from 1 and onward because 0th index is atomic symbol
    a = np.array(geom[plane_a_indices[0]][1:])
    b = np.array(geom[plane_a_indices[1]][1:])
    c = np.array(geom[plane_a_indices[2]][1:])

    d = np.array(geom[plane_b_indices[0]][1:])
    e = np.array(geom[plane_b_indices[1]][1:])
    f = np.array(geom[plane_b_indices[2]][1:])

    print(a)

    # Construct the same vectors used in the original function
    ab = np.array([a[1] - b[1], a[2] - b[2], a[3] - b[3]], dtype=float)
    bc = np.array([b[1] - c[1], b[2] - c[2], b[3] - c[3]], dtype=float)
    de = np.array([d[1] - e[1], d[2] - e[2], d[3] - e[3]], dtype=float)
    ef = np.array([e[1] - f[1], e[2] - f[2], e[3] - f[3]], dtype=float)

    # Compute normal vectors exactly as in the original function
    n1 = np.cross(ab, bc)
    n2 = np.cross(de, ef)

    # Guard against collinear atoms giving zero normal vectors
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm == 0 or n2_norm == 0:
        logger.error('Unable to compute plane angle for %s because one plane is degenerate.', file.name)

    # Compute the raw plane angle
    cos_theta = np.dot(n1, n2) / (n1_norm * n2_norm)

    # Clamp for numerical stability (removes issues with floating point arithmetic)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Round to 3 decimal points
    raw_angle = round(float(np.degrees(np.arccos(cos_theta))), 3)

    # Force selection of smallest angle regardless of sign
    plane_angle = min(abs(raw_angle), abs(180 - raw_angle))

    results = {f'plane_angle_{"-".join(plane_a_names)}_{"-".join(plane_b_names)} (º)': plane_angle}

    #except Exception as e:
    #    logger.error('Could not compute plane angle for %s because %s', file.name, e)

    return pd.DataFrame(pd.Series(results))


def get_plane_angle(dataframe: pd.DataFrame,
                    plane_atom_names: list[list[str]],
                    data_dir: Path,
                    procs: int = 1) -> pd.DataFrame:
    '''
    Gets the distance between atoms defined by a list of list of strings
    where the innermost strings are the atom labels in the dataframe
    (dist_list, form [[C1, O2], [C4, C1]]) in a dataframe that contains
    <FILE_COLUMN_NAME> and atom number.

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    dist_list: list[list[str]]
        List of lists that contain the pairs of atom labels for which
        the distance will be calculated.

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''

    '''
    Gets the nbo for all atoms (a_list, form ["C1", "C4", "O2"]) in a
    dataframe that contains file name and atom number (1-indexed)

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing <FILE_COLUMN_NAME> column

    data_dir: Path
        Directory where the files are located

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the <FILE_COLUMN_NAME> column and
        the resultant descriptors
    '''

    if len(plane_atom_names) != 2:
        raise ValueError(f'You must specify only 2 list of 3 atom names to compute plane angles.')

    # Get just the df of the filename and the atoms needed for plane angle calculations
    interesting_columns = [FILE_COLUMN_NAME]
    interesting_columns.extend(list(set([x for xs in plane_atom_names for x in xs])))
    calculation_df = dataframe[interesting_columns].copy(deep=True)

    # Convert the <FILE_COLUMN_NAME> column to path
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]

    # Get the rows of the dataframe that we will use as input for parallelization
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_plane_angle,
                            zip(calculation_rows,
                                itertools.repeat(plane_atom_names[0]),
                                itertools.repeat(plane_atom_names[1])))

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    logger.info('Plane angle funciton has completed.')
    return dataframe

if __name__ == "__main__":

    file = Path('./data/test_files/Ac2_1.log')

    with open(file, 'r', encoding='utf-8') as infile:
        text = infile.read()

    fake_row = pd.Series({FILE_COLUMN_NAME: 'Ac2_1.log'})
    fake_data_dir = Path('/.data/test_files/')

    _get_plane_angle(file=file,
                     plane_a_indices=[1, 2, 13],
                     plane_b_indices=[1, 3, 13])
