## Usage

Use the gpenv_311.yml file to create a conda environment. You must also install the old version of GoodVibes

## Installation

1.  Create a conda environment using the  gpenv_311.yml file. <br>
    `conda env create -f gpenv_311.yml`
2.  Activate the environment. <br>
    `conda activate gp_env`
3.  Install Goodvibes (Jupyter Notebook branch) <br>
    `git clone https://github.com/patonlab/goodvibes` <br>
    `cd goodvibes` <br>
    `git checkout GV2021` <br>
    `python setup.py install` <br>

## Parallelized Functions
1.  get_goodvibes_e
2.  get_frontierorbs
3.  get_polarizability
4.  get_dipole
5.  get_volume
6.  get_SASA
7.  get_nbo
8.  get_nmr
9.  get_distance
10. get_angles
11. get_dihedral
12. get_vbur_scan
13. get_sterimol_morfeus
14. get_chelpg
15. get_hirshfeld
16. get_pyramidalization

# Non-parallelized Functions
1.  get_time
2.  get_IR
3.  get_sterimol_dbstep
4.  get_sterimol2vec
5.  get_planeangle

# To-do
1. Implement get_IR function

## Major changes
- `get_buried_sterimol` has been combined into `get_sterimol_morfeus`. Specifying a radius will automatically bury the molecule while regular sterimol is calculated if `radius` is `None`
- Logfiles are automatically converted if a .mol file is with the same file stem is not found
- Logfiles can be in any directory instead of the same directory as the Jupyter notebook
- Substructure atom mapping now includes automatic plotting to help in the assignment of atom labels to the substructure atom numbers