#!/usr/bin/env python3
# coding: utf-8

'''
Utility functions for getting properties
'''

import PIL
import subprocess
from pathlib import Path
from io import StringIO

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Chem.Draw import rdMolDraw2D

FILE_COLUMN_NAME = 'file'

def log_to_sdf(log: Path):
    subprocess.run(args=['obabel', '-ilog', f'{log.absolute()}', '-osdf', '-m'], cwd=log.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def log_to_mol_file(log: Path):
    subprocess.run(args=['obabel', '-ilog', f'{log.absolute()}', '-omol', '-m'], cwd=log.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def _get_atom_map(file: Path,
                  substructure: Chem.Mol,
                  debug: bool = False) -> list | tuple[Path, None]:
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
    if file.suffix not in ['.sdf', '.mol']:
        raise ValueError(f'Atom maps can only be extracted from .sdf files not {file.name}')

    if debug:
        print(f'[DEBUG] Reading in {file.name} for atom map')

    mapping = []
    logfile = file.with_suffix('.log')

    if not logfile.exists():
        print(f'[WARNING] Can not locate {logfile.absolute()}')

    mapping.append(logfile.name)

    # Read in the sdf file
    file, mol = _read_in_mol_sdf_with_xyz_correction(file)
    if mol is None:
        return file, None

    # Get the substructure match
    match = mol.GetSubstructMatches(substructure)

    # Check that there is only one substructure
    if len(match) == 0:
        print(f'[ERROR] No matching pattern for {file.name}')
        return file, None
    elif len(match) > 1:
        print(f'[ERROR] More than one matching pattern for {file.name} {match}')
        return file, None

    # Append the atom number (1-indexed values) of the substructure match
    for idx in match[0]:
        mapping.append(idx + 1)

    return mapping

def get_filecont(log: str | Path, no_splitting: bool = False) -> list[str] | str: #gets the entire job output
    error = "" #default unless "normal termination" is in file
    an_error = True

    if isinstance(log, str):
        if '.log' not in log:
            log = log + '.log'
        log = Path(log)
    if not log.exists():
        raise FileNotFoundError(f'{log.absolute()} does not exist.')

    with open(log, 'r', encoding='utf-8') as f:
        loglines = f.readlines()
    for line in loglines[::-1]:
        if "Normal termination" in line:
            an_error = False
        if an_error:
            error = f"****Failed or incomplete jobs for {log.name}"

    if no_splitting:
        with open(log, 'r', encoding='utf-8') as infile:
            text = infile.read()
        return text, error

    return(loglines, error)

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

def mol_to_image(mol, show_atom_indices: bool = True) -> str:
    '''
    Draws an RDKit mol and returns a SVG image.
    '''

    if show_atom_indices:
        for atom in mol.GetAtoms():
            # For each atom, set the property "atomNote" to a index+1 of the atom
            atom.SetProp("atomNote", str(atom.GetIdx()))
    d2d = rdMolDraw2D.MolDraw2DSVG(400, 400)
    dopts = d2d.drawOptions()
    dopts.bondLineWidth = 5
    dopts.baseFontSize = 3.5
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

if __name__ == "__main__":
    # Testing the bond order assignment from rdkit
    file = Path('./data/11171210_2.mol')

    mol = Chem.MolFromMolFile(str(file.absolute()))

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
        print(f'[WARNING] Could not read {file.name} because None mol. Reading in as .xyz')
        if file.suffix == '.mol':
            proc = subprocess.run(args=['obabel', '-imol', f'{file.absolute()}', '-oxyz'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif file.suffix == '.sdf':
            proc = subprocess.run(args=['obabel', '-isdf', f'{file.absolute()}', '-oxyz'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if '1 molecule converted' not in proc.stderr.decode('utf-8'):
            print(f'[ERROR] Could not read in {file.name} as .xyz or as {file.suffix}')
            return file, None

        mol = Chem.MolFromXYZBlock(proc.stdout.decode('utf-8'))

        if mol is None:
            print(f'[ERROR] Could not read in {file.name} as .xyz or as {file.suffix}')
            return file, None

        try:
            rdDetermineBonds.DetermineConnectivity(mol)
            rdDetermineBonds.DetermineBondOrders(mol)
        except ValueError as e:
            print(f'[WARNING] Could not determine bond orders for {file.name}')
            return file, None

    return file, mol