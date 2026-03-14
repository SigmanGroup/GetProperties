#!/usr/bin/env python3
# coding: utf-8

'''
Utility functions for getting properties
'''
import sys
import logging
import subprocess
import multiprocessing
from pathlib import Path
from io import StringIO

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol

import py3Dmol

logger = logging.getLogger(__name__)

FILE_COLUMN_NAME = 'file'

def log_to_sdf(log: Path):
    subprocess.run(args=['obabel', '-ilog', f'{log.absolute()}', '-osdf', '-m'], cwd=log.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def log_to_mol_file(log: Path):
    subprocess.run(args=['obabel', '-ilog', f'{log.absolute()}', '-omol', '-m'], cwd=log.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def _get_atom_map(file: Path,
                  substructure: Chem.Mol) -> list | tuple[Path, None]:
    '''
    Gets an atom mapping from a .sdf file based on
    another substructure

    Parameters
    ----------
    file: Path
        An .mol or .sdf file that has a corresponding
        Gaussian16 logfile in the same directory

    substructure: Chem.Mol
        Mol object that represents the substructure to be
        mapped

    Returns
    ----------
    list
        List containing the name of the logfile and the
        atom numbers (1-indexed) of the mapped structure.
    '''
    configure_logger(debug=True)
    if file.suffix not in ['.sdf', '.mol']:
        raise ValueError(f'Atom maps can only be extracted from .sdf files not {file.name}')

    mapping = []
    logfile = file.with_suffix('.log')

    if not logfile.exists():
        logger.error('Could not locate %s', logfile.absolute())
        return file, None

    mapping.append(logfile.name)

    # Read in the sdf file
    file, mol = _read_in_mol_sdf_with_xyz_correction(file)
    if mol is None:
        return file, None

    # Get the substructure match
    match = mol.GetSubstructMatches(substructure)

    # Check that there is only one substructure
    if len(match) == 0:
        logger.error('No matching pattern for %s', file.name)
        return file, None
    elif len(match) > 1:
        logger.error('More than one matching pattern for %s\t%s', file.name, match)
        return file, None

    # Append the atom number (1-indexed values) of the substructure match
    for idx in match[0]:
        mapping.append(idx + 1)

    return mapping


def get_filecont(log: str | Path,
                 split: bool = True) -> list[str] | str:

    # default unless "normal termination" is in file
    error = ''

    # Correct
    if isinstance(log, str):
        if '.log' not in log:
            log = log + '.log'
        log = Path(log)

    if not log.exists():
        raise FileNotFoundError(f'{log.absolute()} does not exist.')

    # Read in the text
    with open(log, 'r', encoding='utf-8') as infile:
        text = infile.read()

    # Check for simple termination errors
    if 'Error termination' in text:
        error = f'****Failed or incomplete jobs for {log.name}'
    if 'Normal termination' not in text:
        error = f'****Failed or incomplete jobs for {log.name}'

    # The caller requests to split the lines
    if split:
        with open(log, 'r', encoding='utf-8') as infile:
            return infile.readlines(), error

    return text, error

def get_outstreams(log: Path | str): #gets the compressed stream information at the end of a Gaussian job
    streams = []
    starts,ends = [],[]
    error = ""
    an_error = True

    if isinstance(log, Path):
        pass
    elif isinstance(log, str):
        if '.log' in log:
            pass
        else:
            log = log + '.log'
        log = Path(log)

    if not log.exists():
        raise FileNotFoundError(f'{log.absolute()} does not exist.')

    with open(log, 'r') as infile:
        loglines = infile.readlines()

    for line in loglines[::-1]:
        if "Normal termination" in line:
            an_error = False
        if an_error:
            error = "****Failed or incomplete jobs for " + log + ".log"

    for i in range(len(loglines)):
        if "1\\1\\" in loglines[i]:
            starts.append(i)
        if "@" in loglines[i]:
            ends.append(i)
    #    if "Normal termination" in loglines[i]:
    #        error = ""

    if len(starts) != len(ends) or len(starts) == 0: #probably redundant
        error = f'****Failed or incomplete jobs for {log.name}'
        return streams, error

    for i in range(len(starts)):
        tmp = ""
        for j in range(starts[i],ends[i]+1,1):
            tmp = tmp + loglines[j][1:-1]
        streams.append(tmp.split("\\"))

    return streams, error


def get_geom(streams):
    # Extracts the geometry from the compressed stream
    geom = []
    for item in streams[-1][16:]:
        if item == "":
            break
        geom.append([item.split(',')[0], float(item.split(',')[-3]), float(item.split(',')[-2]), float(item.split(',')[-1])])
    return geom


def get_specdata(atoms, prop) -> list:
    '''
    input a list of atom numbers of interest and a
    list of pairs of all atom numbers and property
    of interest for use with NMR, NBO, possibly
    others with similar output structures
    '''
    propout = []
    for atom in atoms:
        if atom.isdigit():
            a = int(atom) - 1
            if a <= len(prop):
                propout.append(float(prop[a][1]))
            else:
                continue
        else:
            continue
    return propout


def mol_to_image(mol,
                 show_atom_indices: bool = True,
                 image_size: tuple[int, int] = (400, 400)) -> str:
    '''
    Draws an RDKit mol and returns a SVG image.
    '''

    if show_atom_indices:
        for atom in mol.GetAtoms():
            # For each atom, set the property "atomNote" to a index+1 of the atom
            atom.SetProp("atomNote", str(atom.GetIdx()))
    d2d = rdMolDraw2D.MolDraw2DSVG(image_size[0], image_size[1])
    dopts = d2d.drawOptions()
    dopts.bondLineWidth = 5
    dopts.baseFontSize = 3.5
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()


def _read_in_mol_sdf_with_xyz_correction(file: Path) -> tuple[Path | None]:
    '''
    Reads an RDKit Mol object from an .sdf or .mol file, falling back to an .xyz
    conversion with OpenBabel and reconstructing connectivity and bond orders
    when the initial read fails.

    Parameters
    ----------
    file: Path
        Path to the input structure file. Supported extensions are .sdf and .mol.

    Returns
    -------
    file_and_mol: tuple
        Tuple of (file, mol), where file is the original Path and mol is the
        RDKit Mol object if successfully read, or None if reading and correction
        both fail.
    '''
    if file.suffix == '.sdf':
        mol = next(Chem.SDMolSupplier(str(file.absolute()), removeHs=False))
    elif file.suffix == '.mol':
        mol = Chem.MolFromMolFile(str(file.absolute()), removeHs=False)
    else:
        raise TypeError(f'File {file.name} does not have a supported extension.')

    if mol is None:
        logger.warning('Could not read %s because None mol. Falling back to .xyz', file.name)

        if file.suffix == '.mol':
            proc = subprocess.run(args=['obabel', '-imol', f'{file.absolute()}', '-oxyz'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif file.suffix == '.sdf':
            proc = subprocess.run(args=['obabel', '-isdf', f'{file.absolute()}', '-oxyz'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if '1 molecule converted' not in proc.stderr.decode('utf-8'):
            logger.error('Could not read in %s as .xyz or %s', file.name, file.suffix)
            return file, None

        mol = Chem.MolFromXYZBlock(proc.stdout.decode('utf-8'))

        if mol is None:
            logger.error('Could not read in %s as .xyz or %s', file.name, file.suffix)
            return file, None

        try:
            rdDetermineBonds.DetermineConnectivity(mol)
            rdDetermineBonds.DetermineBondOrders(mol)
        except ValueError as e:
            logger.warning('Could not determine bod orders for %s', file.name)
            return file, None

    return file, mol


def convert_files_in_directory(directory: Path) -> list[Path]:
    '''
    Convert `.log` files in a directory to `.mol` files and identify failures.

    Parameters
    ----------
    directory: Path
        Directory containing `.log` files to convert.

    Returns
    -------
    failed_files: list[Path]
        List of `.log` files whose corresponding `.mol` files were not created.
    '''

    # Find the log files that do not have corresponding mol files
    files_to_convert = [x for x in directory.glob('*.log') if not x.with_suffix('.mol').exists()]

    # Report how many files will be converted
    logger.info(f'Converting {len(files_to_convert)} logfiles to .mol files.')

    # Convert the files in parallel
    if len(files_to_convert) != 0:
        with multiprocessing.Pool() as pool:
            pool.map(log_to_mol_file, files_to_convert)

    # Store files that failed conversion
    failed_files = []

    # Check that each .log has a corresponding .mol file
    for file in files_to_convert:
        if not file.with_suffix('.mol').exists():
            logger.error(f'{file.with_suffix(".mol").absolute()} does not exist.')
            failed_files.append(file)

    # Warn if any files failed conversion
    if failed_files:
        logger.warning(f'{len(failed_files)} logfile conversions failed.')

    return failed_files


def draw_3D_mol(mol: Mol,
                viewport_size: tuple[int, int] = (400, 300)) -> None:
    '''
    Draw an interactive 3D molecule in a Jupyter notebook with 1-indexed atom labels.

    Parameters
    ----------
    mol: Mol
        RDKit molecule with a 3D conformer.

    viewport_size: tuple[int, int]
        Width and height of the viewer in pixels.

    Returns
    -------
    None: None
        Displays the viewer in the notebook.
    '''

    # Create the 3D viewer
    view = py3Dmol.view(width=viewport_size[0], height=viewport_size[1])

    # Add the molecule from an RDKit mol block
    view.addModel(Chem.MolToMolBlock(mol), 'mol')
    view.setStyle({'stick': {}})

    # Get the conformer for atom coordinates
    conf = mol.GetConformer()

    # Add custom 1-indexed labels at each atom position
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)

        view.addLabel(
            str(idx + 1),
            {
                'position': {'x': pos.x, 'y': pos.y, 'z': pos.z},
                'fontColor': 'black',
                'fontSize': 14,
                'showBackground': False
            }
        )

    # Fit the molecule in the viewport
    view.zoomTo()
    view.show()


def split_compound_name(file: str | Path,
                        delimiter: str,
                        return_key: int = 0) -> str:
    '''
    Helper function that splits a file stem into separate parts based
    on a delimiter and returns the portion of the split string based on
    the return key.

    Parameters
    ----------
    file: str | Path
        Path to the file or just the filename

    delimiter: str
        Character on which we will split the name

    return_key: int
        The key of the list of the string split that is returned
    '''
    stem = Path(file).stem
    return str(stem.split(delimiter)[return_key])


def configure_logger(debug: bool = False) -> None:
    '''
    Configure logging for the current process.

    Parameters
    ----------
    debug: bool
        If True, use DEBUG level. Otherwise use INFO level.

    Returns
    -------
    None
        Configures the root logger for the current process.
    '''

    # Configure logging for this process
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
        datefmt='%m/%d/%Y:%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

if __name__ == "__main__":

    # Testing the bond order assignment from rdkit
    file = Path('./data/11171210_2.mol')

    mol = Chem.MolFromMolFile(str(file.absolute()))

