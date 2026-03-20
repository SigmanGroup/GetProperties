# GetProperties

This repository contains an updated version of the Sigman group's "Get_Properties" script with parallelization. For the
original script, please visit the [original repository](https://github.com/SigmanGroup/Get_Properties).

## Installation

1. Create a conda environment using the  gpenv_312.yml file.

    `conda env create -f gpenv_312.yml`

2. Activate the environment and use the notebook.

    `conda activate gp_env`

## Usage

Use the notebook `get_properties.ipynb` to collect your properties. Be sure to set your Jupyter kernel to `gpenv_312`.
Follow the instructions for each cell. Several comparisons between the old and new versions are available in the
`/tests/` directory.

## Parallelized Functions

1. get_goodvibes_data
2. get_frontierorbs
3. get_polarizability
4. get_dipole
5. get_volume
6. get_SASA
7. get_nbo
8. get_nmr
9. get_distance
10. get_angles
11. get_dihedral
12. get_vbur_scan
13. get_sterimol_morfeus
14. get_chelpg
15. get_hirshfeld
16. get_pyramidalization
17. get_planeangle
18. get_time

## Non-parallelized Functions

1. get_IR
2. get_sterimol_dbstep
3. get_sterimol2vec

## To-do

1. Implement get_IR
2. Implement get_sterimol_dbstep
3. Implement get_sterimol2vec
4. Implement get_enthalpies
5. Complete

## Major changes

- Everything is parallelized
- Using logging module instead of print statements (easily redirected to file)
- Interactive 3D visualization of your files for easy atom name assignment
- `get_buried_sterimol` has been combined into `get_sterimol_morfeus`. Specifying a radius will automatically bury the
molecule while regular sterimol is calculated if `radius` is `None`
- Logfiles are automatically converted to .mol files
- Logfiles can be in any directory instead of the same directory as the Jupyter notebook
- File reading failures are decreased using a `.xyz` correction process
- Failure to read in files notifies user

## Contributors

- James R. Howard, PhD
- Brittany C. Haas, PhD
- Melissa A. Hardy, PhD
- Jordan P. Liles, PhD
