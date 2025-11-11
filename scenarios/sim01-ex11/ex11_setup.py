# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: modflow
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ASR Test Simulation: Phreeqc Example 11 Chemistry
#
# Grid Size: ~4800 ft x 4700 ft
# Grid cells 1.2 ft – 155 ft
# Cells per layer = 1032
# Total cells = 5160
#
# Grid Layers:
#  - Layer 1: heads of all cells specified with CHD
#  - Layer 2: side boundaries set with GHB
#  - Layer 3: side boundaries set with GHB
#  - Layer 4: side boundaries set with GHB
#  - Layer 5: heads of all cells specified with CHD
#
# ASR Well simulated using WEL package
#  - 21 Stress Periods
#
# New stress period when ASR pumping changes and at the start of each month
#  - ~ 10-day time steps
#  - Simulation run time ~25-30 seconds
#
# Simplifying assumptions:
#  - ASR Well injected or extracted at a constant rate of 5 MGD
#  - TDS and temperature of injected water was constant at 150 mg/L and 25C
#  - GHBs were setup using map coverages in GMS.  Heads were pulled from the old KRASR SEAWAT local scale model at a few points along the test model boundary, then the map module was used to interpolate the heads to the boundary cells.  The head contours along the boundaries look a little strange during injection and recovery, but it should be ok for the purposes of the test model
#  - TDS and temperature were assumed to be constant along the model boundaries but varied by model layer based on the KRASR local scale model results
#  - Specific storage was constant for each layer
#
#
# The workflow for this example starts with loading a pre-existing MODFLOW 6 model, then builds the PHREEQC yaml file from scratch based off of the MODFLOW 6 input file parameters.

# %% [markdown]
# # Installation and Setup
#
# Create a custom conda virtual environment can be created using the `environment.yml` file included in this repo.

# %% [markdown]
# ## Python Imports

# %%
import os
import shutil
from pathlib import Path
from importlib import reload

import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
import matplotlib.pyplot as plt
import zipfile

import flopy
from modflowapi import ModflowApi
import phreeqcrm

import utils # from this repo

# %%
# Import the MFRTM package, installed using `conda develop`
import mf6rtm

display(mf6rtm.__file__)
try:
    # if current LimnoTech development version
    display(mf6rtm.__version__)
except AttributeError:
    pass

# %% [markdown]
# NOTE: the notebook below runs from the `externalio` branch the upstream repo 
# (Pablo's) https://github.com/p-ortega/mf6rtm/tree/feature/externalio

# %% [markdown]
# ### If you get `ModuleNotFoundError`
#
# ... you need to install this package into your environment using 
# [`conda develop`]
# (https://docs.conda.io/projects/conda-build/en/latest/resources/commands/conda-develop.html) 
# command in your terminal with your local absolute path to the source directory
#  of this repo. Then restart the kernel.
#
# Note: You can manually added the path to:  
# `~/miniconda3/envs/modflow/lib/python3.12/site-packages/conda.pth`

# %%
# Find your current working directory, which should be folder for this notebook.
working_dir = Path.cwd()
# Find repository path (i.e. the parent to `/examples` directory for this notebook)
repo_path = working_dir.parent
repo_path

# %%
# If MF6RTM is installed in the same directory as this repository,
# then `mf6rtm_source_path` should be the source directory
mf6rtm_source_path = repo_path.parent / "mf6rtm"
if mf6rtm_source_path.exists() == False:
    # If Lauren's laptop
    mf6rtm_source_path = repo_path.parent / "mf6rtm"
print(
    "MF6RTM Source path exists?",
    mf6rtm_source_path.exists(),
    mf6rtm_source_path,
)

# %% [markdown]
# Use the Jupyter [`!` shell command](https://jakevdp.github.io/PythonDataScienceHandbook/01.05-ipython-and-shell-commands.html) to run the `conda develop {source_path}` terminal command directly from this notebook.
#
# NOTE: The Jupyter [`%conda` magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-conda) will note work with `%conda develop {source_path}` in Windows, because it prepends the local working directory to the source path, inserting the wrong path to the `conda.pth` file.

# %%
# Uncomment the line below the first time you run this notebook
# !conda develop {mf6rtm_source_path}

# %% [markdown]
# If the path was added, Restart the kernel and rerun the cells above.
#
# NOTE: if you have already done this once before, you may need to manually 
# delete previous paths and add the path to:
# `~/miniconda3/envs/modflow/lib/python3.12/site-packages/conda.pth`

# %% [markdown]
# ### Alternative to using "conda develop" command to populate conda.pth file
# Adapted from Clearwater Riverine examples\03_01_coupling_riverine_modules_nsm.ipynb

# %%
# Get path for active environment
active_env_path = Path(os.environ['CONDA_PREFIX'])
active_env_path 

# %%
# Find site-packages folder in path for active environment
site_packages_folder = 'site-packages'
paths = []
for site_packages_path in active_env_path.rglob(site_packages_folder): # rglob for recursive search
    paths.append(site_packages_path)
paths

# %%
#create Path object for conda.pth file
conda_pth_filePath = site_packages_path / 'conda.pth'
 
#check if conda.pth file exists
if conda_pth_filePath.exists():
    print('conda.pth file exists, with these contents:')
    # Open the file in read mode ('r')
    with open(conda_pth_filePath, 'r') as file:
        # Read the entire content of the file
        file_contents = file.read()
        print(file_contents)
else:
    conda_pth_filePath.parent.mkdir(parents=True, exist_ok=True)
    with open(conda_pth_filePath, 'a'):
        print('conda.pth file created')

# %%
# add needed path info to conda.pth file, if necessary
with open(conda_pth_filePath, 'a+') as file:
    # --- Read existing content ---
    # To read, you must first move the cursor to the beginning of the file
    file.seek(0)
    file_contents = file.read()
    if str(mf6rtm_source_path) in file_contents:    
        print(f'conda.pth file already includes {mf6rtm_source_path}')
    else:
    # --- Append new content ---
        file.seek(0)
        file.write(str(mf6rtm_source_path))
        file.write('\n')
        print(f'conda.pth file has been modified by adding {mf6rtm_source_path}')

# %% [markdown]
# ## Set Paths to Input and Output Files with `pathlib`
#
# Use the [pathlib](https://docs.python.org/3/library/pathlib.html) library 
# (built-in to Python 3) to manage paths indpendentely of OS or environment. 
# See this [blog post]
# (https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f) 
# to learn about the many benefits over using the `os` library.

# %% [markdown]
# ### Modflow Inputs

# %%
simulation_name = "ASR_DISV_Ex11"

# Path to MF6 inputs created by NAP
sim_ws = Path(
    # Default if working from this repository
    working_dir / simulation_name
)
# Check to see if model directory exists. If it does, delete it and start fresh
zip_path = sim_ws.parent / f"{simulation_name}.zip"
extract_to = sim_ws.parent

if sim_ws.is_dir():
    print("Model files already exist. Removing and unzipping original files.")
    shutil.rmtree(sim_ws)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
else:
    print("No prior model found. Unzipping original files.")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

if sim_ws.exists():
    print("sim_ws exists")
else:
    print("project path does not exist")

# Set simulation workspace equal to input path
mf6_input_path = sim_ws

# %%
# Set filepath for the MF6 simulation configuration file
sim_nam_file_path = sim_ws / "mfsim.nam"
sim_nam_file_path.exists()

print("MF6 sim file exists?", sim_nam_file_path.exists())
sim_nam_file_path

# %%
# # Set the simulation name
# sim_name = "test7_wel_ex11"  # for figure only

# # to save the edited simulation model files as a separate simulation, specify name here:
# new_sim_name = None
# if new_sim_name != None:
#     new_sim_path = working_dir / new_sim_name
#     new_flow_output_dir = new_sim_path / "flow_output"
#     new_flow_output_dir.mkdir(parents=True, exist_ok=True)
#     new_TDS_output_dir = new_sim_path / "trans-TDS_output"
#     new_TDS_output_dir.mkdir(parents=True, exist_ok=True)
#     new_temp_output_dir = new_sim_path / "trans-temp_output"
#     new_temp_output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### PHREEQC Inputs

# %%
# Phreeqc input file folder
chem_inputs_path = working_dir / "chem_inputs_ex11"
if chem_inputs_path.exists():
    chem_input_files = os.listdir(chem_inputs_path)
else:
    print("chem_inputs_path does not exist")
    chem_inputs_path.mkdir(parents=True, exist_ok=True)
chem_input_files

# %%
# Copy input files to simulation workspace directory (i.e. project path)
for file in chem_input_files:
    shutil.copy2(chem_inputs_path / file, sim_ws)

# %%
# Path to PHREEQC Block Input CSV Files
solutions_filepath = sim_ws / "chem_solutions.csv"
exchanges_filepath = sim_ws / "chem_exchanges.csv"
print("PHREEQC input files exist?", solutions_filepath.exists(), exchanges_filepath.exists(), )

# %%
# Path to PhreeqcRM YAML simulation configuration file
# Which we will create below
#  Path to PhreeqcRM YAML created by mf6trm
phreeqc_mf6rtm_yaml_filepath = sim_ws / "mf6rtm.yaml"
print("MF6RTM-created PHREEQCRM YAML file exists?", phreeqc_mf6rtm_yaml_filepath.exists())

# %%
# Set path to PHREEQC Input file (*.pqi)
phreeqc_input_file = "phinp.dat" # created by mf6rtm.mup3d chem_units = "mol/kgw"
postfix_filepath = sim_ws /  'ex4_postfix.phqr'

phreeqc_input_filepath = sim_ws / phreeqc_input_file
print("PHREEQC input file exists?", phreeqc_input_filepath.exists(), postfix_filepath.exists())

# %%
# Select PHREEQC database file
# phreeqc_database_file = "phreeqc.dat" # used in Ex6?
phreeqc_database_file = 'pht3d_datab.dat' # used in Ex4
phreeqc_databases_path = mf6rtm_source_path / "benchmark" / "database"
phreeqc_database_filepath = phreeqc_databases_path / phreeqc_database_file
print("PHREEQC database file exists?", phreeqc_database_filepath.exists())

# %% [markdown]
# ## Set Path to MF6 Executable & Library
# Different versions can be downloaded from: https://github.com/MODFLOW-ORG/executables to a folder similar to this: `bin/mf6.5.0/macarm` 
#
# On Mac, will need to give permissions with these terminal commands from the 
# ```sh
# xattr -dr com.apple.quarantine mf6
# xattr -dr com.apple.quarantine libmf6.dylib
# ```
#

# %%
use_version_installed_with_modflowapi = False
# user = "Laren"
user = "Anthony"
# version = "6.4.2"
version = "6.5.0"
os = "macarm"

try:
    mf6_exe = "mf6"
    dll = "libmf6"
    mf6_version = !{mf6_exe} --version
    mf6dll_version = ModflowApi("libmf6").get_version()
    print(f"Executable & library installed with modflowapi: {mf6_version[1]}, dll: {mf6dll_version}")
except Exception:
    print("Modflow executables not found in environment")

if use_version_installed_with_modflowapi:
    print(f"Using executable installed with modflowapi: {mf6_version[1]}")
else:
    if user == "Lauren":
        # If using executable from GMS
        mf6_bin_path = Path(r"C:/program files/gms 10.8 64-bit/python/lib/site-packages/xms/executables/modflow6")
        mf6_exe = mf6_bin_path / "mf6.exe"
        dll = mf6_bin_path / "libmf6.dll"
    elif user == "Anthony":
        mf6_bin_path = repo_path / "bin" / f"mf{version}" / os
        mf6_exe = mf6_bin_path / "mf6"
        dll = mf6_bin_path / "libmf6.dylib"
    else:
        print("Create a new user and set paths to mf6 and libmf6")

mf6_version = !{mf6_exe} --version
mf6dll_version = ModflowApi(dll).get_version()
print(f"User-selected executable ({mf6_exe.exists()}): {mf6_version[1]}, dll: {mf6dll_version}")

# %%
# Copy executable and library to simulation workspace directory (i.e. project path)
shutil.copy2(mf6_exe, sim_ws)
shutil.copy2(dll, sim_ws)

# %%
(sim_ws/mf6_exe.name).exists()

# %% [markdown]
# ## Utility Functions
#
# These functions moved to `LOWRP_ASR/utils.py`:
# - `run_models()`
# - `write_models()`
# - `create_mf6_gwt()`
# - `get_times_c()`
# - `get_concentrations()`
# - `convert_molL_gL()`
# - `convert_molL_kgft3()`
# - `modify_wel_spd()`

# %% [markdown]
# # Load Modflow 6 Simulation for Reactive Transport
# To set up PhreeqcRM and MF6RTM simulation objects.
# Modifies Modflow 6 input files to include transport models for all reactive transport species.

# %%
# Load simulation using Flopy
sim = flopy.mf6.MFSimulation.load(
    sim_ws=sim_ws,
    exe_name=mf6_exe,  #'mf6',
    verbosity_level=0,
)
sim.model_names

# %% [markdown]
# ## Modify Flow Model

# %%
# load existing gwf model
for model_name in sim.model_names:
    model = sim.get_model(model_name)
    if model.model_type == "gwf6":
        gwf = model
        print(gwf.name)
gwf.get_package_list()

# %%
# removes buy package from gwf model
gwf.remove_package("buy")
gwf.get_package_list()

# %%
# modify output control package to not print head to .lst file
oc = gwf.get_package("oc")
print_record = oc.printrecord.get_data()
print_rec = print_record[0]
mask = ~(
    (print_rec.rtype == "head")
    & (print_rec.ocsetting == "all")
    & (print_rec.ocsetting_data == None)
)
print_record_new = {}
print_record_new[0] = print_record[0][mask]
oc.printrecord.set_data(print_record_new)

# %%
# modify npf package to save specific discharge
npf = gwf.get_package("npf")
npf.save_specific_discharge = True

# %% [markdown]
# ## Read Grid Info

# %%
# Get groundwater model names and grid info
for model_name in sim.model_names:
    # Collect model info 
    model = sim.get_model(model_name)
    model_type = model.model_type
    grid_type = model.get_grid_type()
    grid_units = model.modelgrid.units
    # Collect grid information
    grid_package = model.get_package(grid_type.name)
    nlay = grid_package.nlay.get_data()  # number of layers
    ncpl = grid_package.ncpl.get_data()  # number of cells per layer
    print(f"{model_name}: ", model_type, grid_type.name, grid_units, nlay, ncpl)

# %%
# Use spatial discretization info from the last model
# Calculate total number of grid cells
nxyz = nlay * ncpl
nxyz

# %%
# Get groundwater flow model object
# defined above
display(gwf.name, gwf.get_package_list())

# %% [markdown]
# ### Cell Spacing

# %%
grid_package.length_units

# %%
celldata = grid_package.cell2d.get_data()
# data stored in numpy record arrays  
# can be easily converted to pandas dataframes
cells_df = pd.DataFrame.from_records(celldata, index='icell2d')
cells_df

# %%
verticedata = grid_package.vertices.get_data()
vertices_df = pd.DataFrame.from_records(verticedata, index='iv')
vertices_df

# %%
domain_size = vertices_df.max() - vertices_df.min()
domain_size

# %% [markdown]
# ## Read Time Step Info

# %%
# Get time discretization info from the `tdis` package
tdis = sim.tdis
nper = tdis.nper.get_data()          # number of stress periods
perioddata = tdis.perioddata.get_data()
nstp = perioddata['nstp']            # number of timesteps per stress period
perlen = perioddata['perlen']        # length of stress periods
tsmult = perioddata['tsmult']        # timestep multiplier
t_units = tdis.time_units.get_data() # units

print(f'{nper} stress periods. Units: {t_units}')
print(nstp)
print(perlen)
print(tsmult)

# %% [markdown]
# ## Read Stress Period Info

# %%
# boundary conditions for chem_stress
# get boundary condition packages with transport
# read in spd stress period data ... which varries based on package...

# for well package (wel):
wel = gwf.get_package('wel')
if wel.has_stress_period_data == True:
    spd_wel_dict = wel.stress_period_data.get_data(full_data=True) # full data is default
display(spd_wel_dict)

# %%
# data stored in a dictionary of numpy record arrays, 
# each of which can be easily converted to a pandas dataframe
stress_period_id = 3
pd.DataFrame.from_records(spd_wel_dict[stress_period_id])

# %%
# flopy has a convenient dataframe interface 
# that also includes, the auxilary data (i.e. components) when present
spd_wel_df_dict = wel.stress_period_data.dataframe
spd_wel_df_dict[stress_period_id]

# %%
# ... which allows easy concatination into a single dataframe
spd_wel_df = pd.concat(spd_wel_df_dict.values(), keys=spd_wel_df_dict.keys())
spd_wel_df = spd_wel_df.droplevel(level=1)
# spd_wel_df.index.set_names(['stress_period_id'], inplace=True)
spd_wel_df.info()
spd_wel_df

# %% [markdown]
# NOTE: `cellid` is the cell identifier, and depends on the type of grid that is used for the simulation. 
# - For a structured grid that uses the DIS input file, CELLID is the layer, row, and column. 
# - For a grid that uses the DISV input file, CELLID is the layer and CELL2D number. 
# - If the model uses the unstructured discretization (DISU) input file, CELLID is the node number for the cell.

# %% [markdown]
# # Get Geochemistry for Transport Models and their Initial Conditions
#
# This first step is to create MF6 Groundwater Transport Models (GWT) for each transportable geochemical component, including setting initial conditions (IC).
#
# This requires running an initial PHREEQC calculation from measured inputs, using utilities from the [`mf6rtm`](https://github.com/p-ortega/mf6rtm) package. 
#
# Our workflow, similar to [`mf6rtm` example 4](https://github.com/p-ortega/mf6rtm/blob/main/benchmark/ex4.ipynb), requires these steps:
# - read inputs by PHREEQC "keyword data blocks"
# - convert to a dictionary
# - instantiate `mup3d.{Block}` classes that contain the block's geochemical components
# - set the grid size/shape for the components
# %% [markdown]
# ### SOLUTION Block
# See PHREEQC3 Manual, page 189

# %%
# Read Geochemical Inputs file
# for aqueous phase ("solution") components
solutions_df = pd.read_csv(solutions_filepath, index_col="component")
solutions_df
# %%
# convert dataframe to a Keyword Data Block dictionary
# NOTE: `mf6rtm.mup3D()` currently assigns block numbers by column, starting at 1
solutions_dict = mf6rtm.utils.solution_df_to_dict(solutions_df)

# add data to the mup3d class
solutions = mf6rtm.mup3d.Solutions(solutions_dict)
solutions.data

# %%
solutions.names

# %% [markdown]
# #### Assign SOLUTION Initial Conditions (IC) to all Grid Cells by Block Number

# %%
# mup3d currently requires a grid array with 3 dimensions
""" conc[0].shape = 
        (240, 2, 1, 80)
        ^     ^  ^  ^
        |     |  |  number of cells per layer (ncpl)
        |     |  dummy row dimension (always 1 for DISV)
        |     number of layers (nlay = 2)
        number of time steps (240)"""
# So assign dummy dimensions
nrow = 1
ncol = ncpl # should equal ncpl, but simplifying for now

# %%
# Assign solution block numbers to each in grid
# NOTE: at this stage of creating modflow transport models (gwt), we only want one cell per block

# start by assigning solution block 1 to all cells
grid_ic_solution_numbers = np.ones((nlay, 1, ncpl), dtype=int)

# Modify block assignments over grid, as needed

solutions.set_ic(grid_ic_solution_numbers)
solutions.ic

# %% [markdown]
# #### Assign SOLUTION Boundary Conditions (BC) to all Inflows by Block Number
# Using the Mup3D.ChemStress class to assign Stress Period Data (SPD)

# %%
# Create a well chemistry object
wellchem = mf6rtm.mup3d.ChemStress('wel')

# Assign solution block number to stress period data (spd)
# TODO: implement for multiple wells?
sol_spd = [2] 
wellchem.set_spd(sol_spd)
wellchem.sol_spd

# %%
# Confirm that stress period data (spd) is properly assigned
for data_column_number in wellchem.sol_spd:
     solutions_list_index = data_column_number - 1
     for key, value in solutions.data.items():
        print(key, value[solutions_list_index])

# %% [markdown]
# ### EXCHANGE Block
#
# See PHREEQC3 Manual, page 189
# %%
# Read Geochemical Inputs file for exchange phase components
exchange_df = pd.read_csv(exchanges_filepath, index_col="component")
exchange_df

# %%
# convert dataframe to a Keyword Data Block dictionary
exchange_dict = {0:exchange_df.T.to_dict(index='component')}

# add data to the mup3d class
exchanger = mf6rtm.mup3d.ExchangePhases(exchange_dict)
exchanger.data

# %%
exchanger.names

# %% [markdown]
# #### Assign EXHANGE Initial Conditions (IC) to all Grid Cells by Block Number

# %%
# Set Solution Block Number for equilibration
# TODO: eliminate need for this by equilibrating to solutions blocks specied over the IC grid
exchanger.set_equilibrate_solutions([1])

# Assign block numbers to each cell
# NOTE: at this stage of creating modflow transport models (gwt), we only want one cell per block
# start by assigning exchange block 0 to all cells
grid_ic_exchange_numbers = np.ones((nlay, 1, ncpl), dtype=int)

exchanger.set_ic(grid_ic_exchange_numbers)

# %%
exchanger.ic

# %% [markdown]
# ### Create a reaction model (RM) instance using the `mf6rtm` `Mup3d` class

# %%
# create model class, with solution initial conditions
reaction_model = mf6rtm.mup3d.Mup3d(simulation_name, solutions, nlay, nrow, ncol)

# set model workspace for saving outputs
reaction_model.set_wd(sim_ws)

# set Phreeqc database
reaction_model.set_database(phreeqc_database_filepath)

# set exchange phases
reaction_model.set_exchange_phases(exchanger)

# set Phreeqc postfix file
reaction_model.set_postfix(postfix_filepath)

print(reaction_model.name, reaction_model.grid_shape)

# %%
reaction_model.solutions.data

# %%
reaction_model.exchange_phases.data

# %% [markdown]
# ### Initialize IC Chemistry over Model Grid 
# This creates a PhreeqcRM instance based on components in Solution Blocks assigned initial conditions over the grid. It then runs a PHREEQC time zero equilibrium calculation for inital speciation.

# %%
# Intializing the mup3d class calculates the equilibrated
# initial concentration array
# NOTE: this is very slow over a large grid. 
# TODO: refactor `solver._get_cdlbl_vect()` to use `np.reshape()`, which is 2x faster. See below.
# Workaround is to just do this for ever solution.
reaction_model.initialize()

# %%
reaction_model.components

# %%
# 1D Array of concentrations (mol/L) structured for PhreeqcRM,
# with each component conc for each grid cell
# ordered by `model.components`
reaction_model.init_conc_array_phreeqc

# %%
# Get component concentrations for selected grid cell
cell_index = 0
ncomps_by_nxyz_conc_array = np.reshape(
    reaction_model.init_conc_array_phreeqc, 
    (len(reaction_model.components), -1),
)
ncomps_by_nxyz_conc_array[:,cell_index]

# %%
# Get component concentrations for a selected grid cell
# converting to units of moles per m^3 (or mmol/L) for modflow
cell_index = 0
ic_df = pd.DataFrame(
    ncomps_by_nxyz_conc_array[:,cell_index] * 1000, # unit conversion
    index=reaction_model.components,
    columns=["initial_conc_mmolL"],
)
ic_df.index.rename("components", inplace=True)
ic_df.index = ic_df.index.astype(pd.CategoricalDtype(ordered=True))
ic_df

# %%
# Dictionary of concentrations in units of moles per m^3 (or mmol/L), 
# and structured to match the shape of Modflow's grid
reaction_model.sconc

# %% [markdown]
# #### Aside to test approaches for reshaping

# %%
# create alias for testing current implementation
c_dbl_vect = reaction_model.init_conc_array_phreeqc

# %%
# # %%timeit
# # Current implementation, using code from `solver._get_cdlbl_vect()`
# [c_dbl_vect[i : i + nxyz] for i in range(0, len(c_dbl_vect), nxyz)]
# # 770 ns ± 10.3 ns

# %%
# # %%timeit
# # Alternate implementation
# np.reshape(reaction_model.init_conc_array_phreeqc, (len(reaction_model.components), -1))
# # 435 ns ± 7.06 ns

# %% [markdown]
# 1.77x faster!

# %% [markdown]
# ### Initialize BC Chemistry for all Inflows

# %%
# Set and initialize stress period chemical concentrations for each well
reaction_model.set_chem_stress(wellchem)

# %%
# Component names
reaction_model.wel.auxiliary

# %%
# Equilbrated concentrations Well 0 boundary conditions (from Solution 2)
# in units of moles per m^3 (or mmol/L)
reaction_model.wel.data

# %%
# Open data for a specifc well as a dataframe
well_id = 0
bc_df = pd.DataFrame(
    reaction_model.wel.data[well_id],
    index=reaction_model.wel.auxiliary,
    columns=["initial_conc_mmolL"],
)
bc_df.index.rename("components", inplace=True)
bc_df.index = ic_df.index.astype(pd.CategoricalDtype(ordered=True))
bc_df

# %%
# Get Modflow's Stress Period Data (spd) from the `wel` package,
# with the well location (cellid), flow rate (q), and other conditions
# as previously collected above
spd_wel_df

# %%
# # Append Conc data to Well Stress Period Data list, 
# # NOTE: only run this once
# for i in range(len(wel_spd)):
#     wel_spd[i].extend(reaction_model.wel.data[i])
# wel_spd

# %% [markdown]
# ### Unit Conversions
#
# - Although MODFLOW is technically agnostic about chemical concentration units used for transport, we have found solver issues when units between transport and reaction models are different.
#
# #### PHREEQC unit handling
# - Although PHREEQC can handle multiple units, all options use the metric system. From PHREEQC3 Manual page 191: 
#   - Three groups of concentration units are allowed, concentration 
#     - (1) per liter (“/L”), 
#     - (2) per kilogram solution (“/kgs”), or 
#     - (3) per kilogram water (“/kgw”). 
#   - All concentration units for a solution must be within the same group. 
#   - Within a group, either grams or moles may be used, and prefixes milli (m) and micro (u) are acceptable. The abbreviations for parts per thousand, “ppt”; parts per million, “ppm”; and parts per billion, “ppb”, are acceptable in the “per kilogram solution” group. 
#   - Default is mmol/kgw.
#
# #### PhreeqcRM unit defaults
# - [`YAMLSetUnitsSolution()`](https://usgs-coupled.github.io/phreeqcrm/namespaceyamlphreeqcrm.html#a6ae20ea754c0f1087ba700dbf48b55a4) uses:
#   - 1, mg/L (default); 
#   - 2 mol/L; or 
#   - 3, mass fraction, kg/kgs.

# %% [markdown]
# # Add Chem to Modflow 

# %% [markdown]
# ## Create MF6 Transport Models for each chemical component
# With initial starting concentrations calculated from initializing PhreeqcRM via the `mup3d.sconc` dictionary, with units of of moles per m^3 (or mmol/L).

# %%
component_name_l = reaction_model.sconc.keys()
component_name_l

# %%
reaction_model.sconc['Na']

# %%
# create new gwt models for each component
porosity = 0.3
dispersivity = 0.00656  # ft = 0.002 # Longitudinal dispersivity (m)
gwf_name = "flow"

for component_name in component_name_l:
    print("Adding gwt model for: " + component_name)
    gwt_name = "trans-" + component_name
    sim = utils.create_mf6_gwt(
        sim, gwf_name, gwt_name, component_name, reaction_model.sconc[component_name], porosity, dispersivity
    )


# %%
# Confirm Modflow models in the simulation
sim.model_names

# %%
# Confirm initial condition concs for Na, from `mup3d.sconc`
# units of moles per m^3 (or mmol/L), 
sim.get_model('trans-Na').ic.strt.array

# %% [markdown]
# ## Add Chem Components to Stress Period Data

# %%
# We created this dataframe from mf6rtm.mup3d inputs
bc_df

# %%
# make aliases for well component names and concentrations
# units of moles per m^3 (or mmol/L), 
component_name_l = reaction_model.wel.auxiliary
wel_conc = reaction_model.wel.data[0]
display(component_name_l, wel_conc)

# %%
# add new components to wel spd and auxvar

# load wel package and stress period data
wel = gwf.wel
spd = wel.stress_period_data.get_data(full_data=True) 
    # NOTE: alsp defined above as `spd_wel_dict`

# modify wel spd data
new_wel_spd = {}
for kper, records in spd.items():
    updated_record = utils.modify_wel_spd(records, component_name_l, wel_conc)
    new_wel_spd[kper] = np.rec.array(updated_record)

# set new aux variables
wel_spd_dtype = list(new_wel_spd[0].dtype.names)
new_wel_auxvar = wel_spd_dtype[2:-1]  # "2:-1" --> excludes wel parameters from auxvars
wel.auxiliary = new_wel_auxvar

# set stress period data to new_wel_spd that includes added components
wel.stress_period_data.set_data(new_wel_spd)


# %%
# Confirm well concentrations, units of moles per m^3 (or mmol/L)
wel.stress_period_data.dataframe[0]

# %%
# Confirm well concentrations, units of moles per m^3 (or mmol/L)
# for every stress period
spd_welchem_df_dict = wel.stress_period_data.dataframe
spd_welchem_df = pd.concat(spd_welchem_df_dict.values(), keys=spd_welchem_df_dict.keys())
spd_welchem_df = spd_welchem_df.droplevel(level=1)
spd_welchem_df

# %%
# modify tdis
tdis_spd = sim.get_package("tdis").perioddata.get_data(full_data=True)
tdis_spd
tdis_spd["nstp"] = tdis_spd[
    "perlen"
]  # set number of steps (nstp) equal to stress period length (perlen) so dt = 1 day for each stress period
# tdis_spd['nstp'][0] = 20 # set first stress period to 20 days with 1 timestep per day
# tdis_spd['perlen'][0] = 20
tdis_spd["nstp"]
sim.get_package("tdis").perioddata.set_data(tdis_spd)

# %% [markdown]
# ## Convergence issues for Lauren
#
# - changed nstp to 10 for each stress period and change ims complexity to moderate --> improved convergence but still failed in sp 3
# - changed ims complexity to complex --> fails at sp 6
# - changed wel q to 0.5 fails
# - flow only and no buy --> runs
# - changed nstp to perlen (dt=1 day) with complex --> failed in sp 2
# - set wel_conc = ic_conc with oringinal tdis, ims = simple, wel_q = 0.5 --> fail sp3
# - set wel_conc = ic_conc with oringinal tdis, ims = complex, wel_q = 0.5 --> normal termination
# - remove tds and buy package, wel_conc = ic_conc, ims = complex, wel_q = 0.5 -->
# - remove tds and buy package, wel_conc = wel_conc, ims = complex, wel_q = 0.5 -->
# - run with just no buy package --> normal termination
# - charge, H, and O removed as solutes... and fixed units --> fail in 2nd sp
# - annnd fixed units in dsp package.... ic = wel_conc, just Na and Ca and tds and temp --> normal termination!! results look ok
# - Ca, Na where ic != wel_conc, tds and temp --> failed to converge in sp2
# - Ca, Na where ic != wel_conc, tds and temp, solver = complex --> normal terminantion
# - try moderate solver option --> normal termination,
# - complex solver with original tdis, ic !=wel_conc --> same with edge patterns

# %%
# Remove transport models for testing
# sim.remove_model('trans-tds')
# sim.remove_model('trans-temp')

# %%
gwt_model_names = [name for name in sim.model_names 
                    if (sim.get_model(name).model_type == 'gwt6')]
print("Number of transport models: ",len(gwt_model_names))
gwt_model_names

# %%
# Confirm Modflow 6 version
# !{mf6_exe} --version

# %% [markdown]
# ## What runs and what crashes with Modflow 6.6.3
#
# These run:
# ```py
# 8: `['trans-tds', 'trans-temp', 'trans-H', 'trans-O', 'trans-Charge', 'trans-Ca', 'trans-Cl', 'trans-K',]`
# 5: `[                           'trans-H', 'trans-O', 'trans-Charge', 'trans-Ca', 'trans-Cl',           ]`
# ```
#
# This run, but end in 14 sec:
# ```py
# 6: `[                           'trans-H', 'trans-O', 'trans-Charge', 'trans-Ca', 'trans-Cl', 'trans-K',]`
# ```
#
# These crash with `Internal error: Too many profiled sections, "increase MAX_NR_TIMED_SECTIONS`:
# ```py
# 7: `[                           'trans-H', 'trans-O', 'trans-Charge', 'trans-Ca', 'trans-Cl', 'trans-K', 'trans-N']`       ]`
# ```

# %%
# write updated simulation input files
sim.write_simulation()

# %% [markdown]
# # Run Modflow 6 simulation only
#
# To confirm that conservative transport is occuring as expected.

# %%
utils.run_models(sim, silent=False)

# %% [markdown]
# ## Plot MF6 Transport Results with no Reactions
#
# When just running MF6, before any coupling

# %%
# lookup cell ID of wel package cell
wel_spd = gwf.wel.stress_period_data.array
wel_cellid = wel_spd[0]["cellid"][0]
wel_cellid
wel_lay = wel_cellid[0]
wel_cellnum = wel_cellid[1]

# %%
# read in results for plots

# head in well cell over time
head = gwf.output.head().get_alldata()
times_h = gwf.output.head().get_times()

# concentration of each component in well cell overtime
conc = utils.get_concentrations(sim, component_name_l)
times_c = utils.get_times_c(sim, component_name_l)

# get specific discharge
bud_flow = gwf.output.budget()
spdis = bud_flow.get_data(text="DATA-SPDIS")


# %%
# plot head
f = 101
fig = plt.figure(num=f, figsize=(18, 5))
plt.plot(times_h, head[:, wel_lay, 0, wel_cellnum], marker=".")
f = f + 1


# %%
# Create list of components to plot based on intersection with transported components
components_to_plot = [c for c in component_name_l if c in ['Ca', 'Cl', 'K', 'N', 'Na']]
components_to_plot

# %%
component_name_l

# %%


k = wel_lay  # layer index
cnum = wel_cellnum  # cell number
for c in range(len(component_name_l)):
    fig = plt.figure(num=101, figsize=(10, 5))
    plt.plot(times_c[c], conc[c][:, k, 0, cnum], label=component_name_l[c])
    plt.title("[" + str(k) + "," + str(cnum) + "]")
    plt.legend()


# # temp and tds gwt output
# temp_tds_l = ["temp", "tds"]
# temp_tds_output = utils.get_concentrations(sim, temp_tds_l)
# times_temptds = utils.get_times_c(sim, temp_tds_l)
# for c in range(len(temp_tds_l)):
#     fig = plt.figure(figsize=(18, 5))
#     plt.plot(times_temptds[c], temp_tds_output[c][:, k, 0, cnum])
#     plt.title(temp_tds_l[c] + " [" + str(k) + "," + str(cnum) + "]")
# # tds and temp plan view figures
# s = 1  # temp_tds_l index
# t_l = [0, 5, 10, 30, 50, -1]  # list of timestep index (NOT actual time/days)
# for t in t_l:
#     fig = plt.figure(figsize=(24, 4))
#     ax = fig.add_subplot(1, 1, 1, aspect="auto")
#     ax.set_title("conc of " + temp_tds_l[s] + " at timestep index t=" + str(t))
#     mapview = flopy.plot.PlotMapView(gwf, layer=3)  # ,extent=(0,0.08,0,1.))
#     patch_collection = mapview.plot_array(
#         temp_tds_output[s][t, :, :, :]
#     )  # ,vmin=26.600, vmax=26.61)
#     linecollection = mapview.plot_grid()
#     cb = plt.colorbar(patch_collection, shrink=0.75)


# xsection
gwf_name = "flow"
gwf = sim.get_model(gwf_name)

# to plot a cross section with disv, you have to make a line to plot along

line = np.array([(694298, 1025429), (6999092, 1025429)])
# creates a plot showing where the line is on the grid to make the cross section plot
fig = plt.figure(figsize=(24, 4))
ax = fig.add_subplot(1, 1, 1, aspect="auto")
ax.set_title("Vertex Model Grid (DISV) with cross sectional line")
# ax.set_xlim(0,0.08)
# ax.set_ylim(0,1.)
# use PlotMapView to plot a DISV (vertex) model
mapview = flopy.plot.PlotMapView(gwf, layer=1)  # ,extent=(0,0.08,0,1.))
# mapview.plot_bc("WEL-1")
# mapview.plot_bc("CHD-1")
linecollection = mapview.plot_grid()
# plot the line over the model grid
lc = plt.plot(line.T[0], line.T[1], "r--", lw=0.8)
plt.show()

# creates a cross section along the line specified above for each timestep in t_l
s = 4  # solute index for Cl
t_l = [0, 1, 10, 25, 50, -1]  # list of timestep index (NOT actual time/days)
normalize = True
if normalize == True:
    scale = 50
else:
    scale = 100
for t in t_l:
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(
        spdis[t], gwf, head=head[t]
    )
    fig = plt.figure(figsize=(18, 5))
    ax = fig.add_subplot(1, 1, 1)
    if normalize == True:
        ax.set_title(
            "normalized specific discharge and conc of "
            + component_name_l[s]
            + " at timestep index t="
            + str(t)
        )
    else:
        ax.set_title(
            "specific discharge and conc of "
            + component_name_l[s]
            + " at timestep index t="
            + str(t)
        )
    xsect = flopy.plot.PlotCrossSection(model=gwf, line={"line": line})
    patch_collection = xsect.plot_array(conc[s][t, :, :, :], vmin=0.0, vmax=1.0)
    line_collection = xsect.plot_grid()
    quiver = xsect.plot_vector(
        qx,
        qy,
        qz,
        head=head,
        hstep=2,
        normalize=normalize,
        color="white",
        scale=scale,  # changes arrow length
        width=0.003,
        headwidth=3,
        headlength=3,
        headaxislength=3,
        zorder=10,
    )
    cb = plt.colorbar(patch_collection, shrink=0.75)
    ## TODO: add a legend for the quiver to relate to spdis magnitude when noralized = False..?

s = 3  # solute index for Ca
t_l = [0, 1, 10, 50, 200, 400, -1]  # list of timestep index (NOT actual time/days)
for t in t_l:
    fig = plt.figure(figsize=(24, 4))
    ax = fig.add_subplot(1, 1, 1, aspect="auto")
    ax.set_title("conc of " + component_name_l[s] + " at timestep index t=" + str(t))
    mapview = flopy.plot.PlotMapView(gwf, layer=2)  # ,extent=(0,0.08,0,1.))
    patch_collection = mapview.plot_array(conc[s][t, :, :, :])  # ,vmin=0., vmax=0.2)
    linecollection = mapview.plot_grid()
    cb = plt.colorbar(patch_collection, shrink=0.75)


# %% [markdown]
# # Reactive Transport Simulation
# Using MF6RTM

# %%
# Run the model using this wrapper function for `mf6rtm.solve(model.wd)`
reaction_model.run()

# %% [markdown]
# ### Initialized PhreeqcRM simulation instance
# Created above and containing all info for reaction module

# %%
# reaction_model?

# %%
# Run the model using this wrapper function for `mf6rtm.solve(model.wd)`
reaction_model.run()

# %%
# mf6rtm.mup3d created PhreeqcRM simulation object
# reaction_model.phreeqc_rm?

# %%
# Access the PhreeqcRM object directly, if necessary
# reaction_model.phreeqc_rm?

# %%
# mf6rtm.mup3d created this PhreeqcRM YAML file
# phreeqc_config_filepath = sim_ws / "mf6rtm.yaml"
reaction_model.phreeqcyaml_file

# %% [markdown]
# ### Initilize ModflowAPI interface

# %%
# initialize the ModflowAPI instance
wd = mf6_input_path
dll = 'libmf6'

mf6 = mf6rtm.Mf6API(wd, dll)

# %%
# mf6?

# %%
# Save list of outputs for use later
output_var_names = mf6.get_output_var_names()
input_var_names = mf6.get_input_var_names()

# %%
# Get list of modflow input varables with concentration data ('/X' or '/XOLD')
gwt_conc_var_names = []
for model_name in gwt_model_names:
    gwt_conc_var_names += [var for var in input_var_names if f'{model_name.upper()}/X' in var]
gwt_conc_var_names

# %%
conc_var_name = gwt_conc_var_names[-1]
conc_var_name

# %%
# Get info from the BMI functions exposed by modflowapi
mf6.get_var_rank(conc_var_name)

# %%
mf6.get_value(conc_var_name)

# %% [markdown]
# ### Initialize & Run MF6RTM coupling interface

# %%
rtm = mf6rtm.Mf6RTM(wd, mf6, reaction_model.phreeqc_rm)

# %% [markdown]
# ### Try this

# %%
reaction_model.wd

# %%
str(mf6_input_path)

# %%
reaction_model.set_wd(str(mf6_input_path))
reaction_model.wd

# %%
reaction_model.run()

# %% [markdown]
# # END
