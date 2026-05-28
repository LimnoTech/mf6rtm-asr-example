# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ASR Simulation 3: Grid 2 (10ft near well) with Chemistry from MF6RTM Example 5 (Appelo 1998) and Arsenic Chemistry from LimnoTech coal ash work.
#
# This simulation adds **arsenic redox chemistry** -- from [MF6RTM Example 5](https://github.com/p-ortega/mf6rtm/blob/main/benchmark/ex5.ipynb) (Appelo 1998) and Arsenic Chemistry from LimnoTech coal ash work -- to the 3D transport models of the simple ASR test case. 
#
# NOTE: THIS EXAMPLE CRASHES with Arsenic.
#
# MF6RTM Example 5 includes these geochemical processes:
# - Solutions (required aqueous chemistry)
# - Exchanges
# - Equilibrium Phases (mineral precipitation/dissolution reaction equilibria)
# - Kinetics (mineral precipitation/dissolution rates)
# - Surfaces (adsorption/desorption onto minerals defined above)
#
# For information and exploration of the simple ASR Modflow 6 simulation used throughout this repository, see `sims/sim00-mf6only/mf6_explore.ipynb`.
#
# The workflow for this example:
# - Read geochemical components and their initial and boundary concentrations from PHREEQC input files
# - Create new Modflow 6 transport model for each aqueous phase (components in the Solution blocks) and add their initial concentrations over the entire DISV grid.
# - Modify the Modflow 6 Flow Well package Stress Period Data (SPD) by adding Solution component concentrations.
# - Run the modified Modflow 6 for conservative transport of all components (i.e. no coupling to PHREEQC)
# - Run the coupled Modflow 6 & PHREEQC models for the entire simulation
#
# NOTE: This [Jupytext](https://jupytext.readthedocs.io/en/latest/index.html) paired notebook, with paired `.py` and `.ipynb` files. 
# - If using VS Code, install the the [Jupytext Sync extension](https://jupytext.readthedocs.io/en/latest/vs-code.html) for maximum benefit.

# %% [markdown]
# # Installation and Setup
#
# Create a custom conda virtual environment can be created using the `environment.yml` file included in this repo. 
#
# Complet the setup by running the `01-GettingStarted.ipynb` notebook.

# %% [markdown]
# ## Python Imports

# %%
import os
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import flopy
from modflowapi import ModflowApi

# %%
# Import the MFRTM package, installed using `conda develop`
import mf6rtm

display(mf6rtm.__file__)
try:
    # if current LimnoTech development version
    display(mf6rtm.__version__)
except AttributeError:
    pass

# %%
from mf6rtm_asr_example import utils # from this repo

# %% [markdown]
# ### If you get `ModuleNotFoundError`
#
# Run the `01-GettingStarted.ipynb` notebook to install `mf6rtm` using `conda develop`.

# %% [markdown]
# ## Set Paths to Executables, Inputs, and Outputs with `pathlib`
#
# Use the [pathlib](https://docs.python.org/3/library/pathlib.html) library 
# (built-in to Python 3) to manage paths indpendentely of OS or environment. 
# See this [blog post](https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f) 
# to learn about the many benefits over using the `os` library.

# %%
# user = "Laren"
user = "Anthony"

# %%
# Find your current working directory, which should be folder for this notebook.
working_dir = Path.cwd()
# Find repository path (i.e. the parent to `/examples` directory for this notebook)
repo_path = working_dir.parent.parent
repo_path
if user == "Laren":
    repo_path = Path("C:\\Users\\rdchllkm\\Documents\\GitHub\\mf6rtm-asr-example")
    working_dir = repo_path / "sims" / "sim03-Wallis2011"
# %%
# Simulation based on chemical inputs
# simulation_name = working_dir.name
simulation_name = "MF6RTM Example 5"
simulation_name

# %%
# Path to simulation workspace, which is git-ignored and 
# will get over-written with each run of this notebook
sim_ws = working_dir / 'ws2xas' # Grid 2
sim_ws.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### MF6 Executable & Library
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
os = "macarm"
# version = "6.4.2"
# version = "6.5.0"
version = "6.7.0"

try:
    mf6_exe = Path(flopy.which("mf6"))
    dll = mf6_exe.parent.parent / "lib" / "libmf6.dylib" # MacOS only for now
    mf6_version = !{mf6_exe} --version
    mf6dll_version = ModflowApi(dll).get_version()
    print(f"Executable & library installed with modflowapi: {mf6_version[1]}, dll: {mf6dll_version}")
except Exception:
    print("Modflow executables not found in environment")

if use_version_installed_with_modflowapi:
    print(f"Using executable installed with modflowapi: {mf6_version[1]}")
else:
    if user == "Lauren":
        # If using executable from GMS
        mf6_bin_path = Path(r"C:\\Users\\rdchllkm\\Documents\\Programs\\mf6.8.0.dev0_win64\\bin")
        mf6_exe = mf6_bin_path / "mf6.exe"
        dll = mf6_bin_path / "libmf6.dll"
    elif user == "Anthony":
        mf6_potential_paths = [
            repo_path / "bin" / f"mf{version}" / os,
            repo_path / "bin" / f"mf{version}_{os}" / "bin",
        ]
        for path in mf6_potential_paths:
            if path.exists():
                mf6_bin_path = path
        mf6_exe = mf6_bin_path / "mf6"
        dll = mf6_bin_path / "libmf6.dylib"
    else:
        print("Create a new user and set paths to mf6 and libmf6")
    mf6_version = !{mf6_exe} --version
    mf6dll_version = ModflowApi(dll).get_version()
    print(f"User-selected executable ({mf6_exe.exists()}): {mf6_version[1]}, dll: {mf6dll_version}")

# %% [markdown]
# ### Reset Workspace & copy files

# %%
# Delete previous contents from simulation
if sim_ws.exists():
    try:
        shutil.rmtree(sim_ws)
        print(f"Directory '{sim_ws}' and its contents removed successfully.")
        sim_ws.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: {sim_ws} : {e.strerror}")
else:
    print(f"Directory '{sim_ws}' does not exist.")

# %%
# Copy modflow executable and library to simulation workspace
shutil.copy2(mf6_exe, sim_ws)
shutil.copy2(dll, sim_ws)
(sim_ws/mf6_exe.name).exists()

# %% [markdown]
# ### Modflow Inputs

# %%
# Modflow inputs file folder
# Grid 2 = 10ft resolution near well (vs 2ft for original)
mf6_inputs_path = repo_path / 'data' / 'MF6_ASR_DISV_inputs2' # Grid 2

# %%
# Copy input files to simulation workspace directory)
shutil.copytree(mf6_inputs_path, sim_ws, dirs_exist_ok=True)

# %%
# Add required empty output folders
folders = [
    'flow_output',
    'trans-TDS_output',
    'trans-temp_output',
]
for folder in folders:
    path = sim_ws / folder
    path.mkdir(parents=True, exist_ok=True)

# %%
# Set filepath for the MF6 simulation configuration file
sim_nam_file_path = sim_ws / "mfsim.nam"
assert sim_nam_file_path.exists()
sim_nam_file_path

# %%
# Set modflow input path equal to simulation workspace
mf6_input_path = sim_ws

# %% [markdown]
# ### PHREEQC Inputs

# %%
# Phreeqc input file folder
chem_inputs_path = working_dir / "chem_inputs_Ex5_As"

chem_prefix = "ex5_"
chem_input_files_match = chem_inputs_path.glob(f"{chem_prefix}*")
chem_input_files = [file.name for file in chem_input_files_match]
chem_input_files

# %%
# Copy input files to simulation workspace directory (i.e. project path)
for file in chem_input_files:
    shutil.copy2(chem_inputs_path / file, sim_ws)

# %%
# Path to PHREEQC Block Input CSV Files
solutions_filepath = sim_ws / f"{chem_prefix}solutions.csv"
exchanges_filepath = sim_ws / f"{chem_prefix}exchanges.csv"
equilibrium_phases_filepath = sim_ws / f"{chem_prefix}equilibrium_phases.csv"
surfaces_filepath = sim_ws / f"{chem_prefix}surfaces.csv"
kinetic_phases_filepath = sim_ws / f"{chem_prefix}kinetic_phases.csv"

assert solutions_filepath.exists() and exchanges_filepath.exists()
assert equilibrium_phases_filepath.exists() and surfaces_filepath.exists()
assert kinetic_phases_filepath.exists()

# %%
# Path to file with PHREEQC Input "postfix" instructions
# to be appended to the PHREEQC Input file (*.pqi) created by mf6rtm
postfix_filepath = sim_ws /  f"{chem_prefix}postfix.phqr"
assert postfix_filepath.exists()

# %%
# Select PHREEQC database file
# phreeqc_database_file = "datab.dat"
# phreeqc_database_file = "phreeqc.dat" # used in Ex5 & 6?
phreeqc_database_file = "phreeqc_As.dat" # modified with As reactions
# phreeqc_database_file = 'pht3d_datab.dat' # used in Ex4
phreeqc_databases_path = repo_path / "data" / "chem_databases"
phreeqc_database_filepath = phreeqc_databases_path / phreeqc_database_file
assert phreeqc_database_filepath.exists(), "PHREEQC database file missing"

# %%
# Paths to PHREEQC configuration files that will be created by mf6trm
# PHREEQC Input file (*.pqi)
phreeqc_input_filepath = sim_ws / "phinp.dat"
# PhreeqcRM YAML config file
phreeqcrm_yaml_filepath = sim_ws / "mf6rtm.yaml"

# %% [markdown]
# # Load Modflow 6 Simulation for Reactive Transport
# To set up PhreeqcRM and MF6RTM simulation objects.
# Modifies Modflow 6 input files to include transport models for all reactive transport species.
#
# For addtional information and exploration of the simple ASR Modflow 6 simulation used throughout this repository, see `sims/sim00-mf6only/mf6_explore.ipynb`

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
# lookup cell ID of wel package cell
# NOTE: indices from flopy use Python indexing from 0
wel_spd = gwf.wel.stress_period_data.array
wel_cellid = wel_spd[0]["cellid"][0]
display(wel_cellid)

wel_lay = wel_cellid[0]
wel_cellnum = wel_cellid[1]

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

# %% [markdown]
# ### Cell Volumes

# %%
### Calculate grid cell volume
cell2D = grid_package.cell2d.get_data()
cell2D_df = pd.DataFrame.from_records(cell2D, index='icell2d')
vertices = grid_package.vertices.get_data()
vertices_df = pd.DataFrame.from_records(vertices, index='iv')
top = grid_package.top.array
botm = grid_package.botm.array

### Calculate surface area of each grid cell within a single layer
for cell in range(len(cell2D_df)):
    ### get vertice coordinates to calc surface area
    temp_cell_info = cell2D_df.iloc[[cell]]
    # read each vert_1 - 5
    icverts = temp_cell_info.filter(like="icvert_").iloc[0].to_list()
    # remove Nones
    icverts_clean = [int(v) for v in icverts if v is not None]
    # look up (x,y) and create x and y arrays
    x_l = vertices_df.loc[icverts_clean,"xv"].to_numpy()
    y_l = vertices_df.loc[icverts_clean,"yv"].to_numpy()
    # calculate surface area based on min/max x/y from array
    surface_area = (np.max(x_l)-np.min(x_l)) * (np.max(y_l) - np.min(y_l))
    cell2D_df.loc[cell,'surface_area'] = surface_area

### Calc layer thickness for each layer
# initialize thickness array
thickness = np.zeros_like(botm)
# for layer 1:
thickness[0] = top - botm[0]
# for layers 2:nlay
thickness[1:] = botm[:-1] - botm[1:]

### Calculate volume for each grid cell
# initialize volume array
cell_volumes = np.zeros((nlay,ncpl))
# calculate volume for entire grid
for k in range(nlay):
    cell_volumes[k,:] = cell2D_df['surface_area'].to_numpy() * thickness[k,:]

# %%
cell2D_df

# %%
# volume of cells near well screen
cell_volumes[2, wel_cellnum-6:wel_cellnum+6]

# %%
# TODO: Get flat Cell Index to (cellid_layer, cellid_cell) mapping
# for exploring phreeqcrm outputs

# %%
cell_flat_index = np.array(range(nlay*ncpl))
cell_flat_index

# %%
cellid_flatmap = np.reshape(cell_flat_index, (nlay,ncpl))
cellid_flatmap

# %%
cellid_layer = wel_lay
cellid_cell = wel_cellnum
cellid_flatmap[cellid_layer, cellid_cell]

# %% [markdown]
# ## Read Time Info

# %% [markdown]
# ### Time Steps

# %%
# Get time discretization info from the `tdis` package
tdis = sim.tdis
nper = tdis.nper.get_data()          # number of stress periods
perioddata = tdis.perioddata.get_data() # record array
nstp = perioddata['nstp']            # number of timesteps per stress period
perlen = perioddata['perlen']        # length of stress periods
tsmult = perioddata['tsmult']        # timestep multiplier
t_units = tdis.time_units.get_data() # units

print(f'{nper} stress periods. Units: {t_units}')
perioddata

# %%
pd.DataFrame.from_records(perioddata)

# %% [markdown]
# ### Stress Period Data

# %%
# boundary conditions for chem_stress
# get boundary condition packages with transport
# read in spd stress period data ... which varries based on package...

# for well package (wel):
wel = gwf.get_package('wel')
if wel.has_stress_period_data == True:
    spd_wel_dict = wel.stress_period_data.get_data(full_data=True) # full data is default
# spd_wel_dict

# %%

# data stored in a dictionary of numpy record arrays
# which allows easy concatination into a single dataframe, using
# flopy dataframe interface that includes auxilary data (i.e. components) when present
spd_wel_df_dict = wel.stress_period_data.dataframe
spd_wel_df = pd.concat(spd_wel_df_dict.values(), keys=spd_wel_df_dict.keys())
spd_wel_df = spd_wel_df.droplevel(level=1)
# spd_wel_df.index.set_names(['stress_period_id'], inplace=True)
spd_wel_df.info()
spd_wel_df

# %% [markdown]
# NOTE: `cellid` is a cell identifier tuple, and depends on the type of grid that is used for the simulation. 
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
solutions_df = pd.read_csv(solutions_filepath, index_col="component", comment = '#',)
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
# #### Assign SOLUTION Boundary Conditions (BC) to Grid Perimeter
# Using the Mup3D.ChemStress class to assign Stress Period Data (SPD)

# %%
# Create a chd chemistry object
chdchem = mf6rtm.mup3d.ChemStress('chd')
chd_sol_spd = [1] # use same solution as inital conditions
# Assign solution block number to stress period data for chd
chdchem.set_spd(chd_sol_spd)
chdchem.sol_spd

# %%
# Create a ghb chemistry object
ghbchem = mf6rtm.mup3d.ChemStress('ghb')
ghb_sol_spd = [1] # use same solution as inital conditions
# Assign solution block number to stress period data for chd
ghbchem.set_spd(ghb_sol_spd)
ghbchem.sol_spd

# %% [markdown]
# #### Assign SOLUTION Boundary Conditions (BC) to all Inflows by Block Number
# Using the Mup3D.ChemStress class to assign Stress Period Data (SPD)

# %%
# Create a well chemistry object
wellchem = mf6rtm.mup3d.ChemStress('wel')

# Assign solution block number to stress period data (spd)
# TODO: implement for multiple wells? See MF6RTM Example 5
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
exchanger.ic

# %% [markdown]
# ### EQUILIBRIUM PHASES Block
#
# See PHREEQC3 Manual, page ??

# %%
#equilibrium phases
equilibriums_df = pd.read_csv(equilibrium_phases_filepath)
equilibriums_df

# %%
equilibriums_dict = mf6rtm.utils.parse_equilibriums_dataframe(equilibriums_df)
equilibrium_phases = mf6rtm.mup3d.EquilibriumPhases(equilibriums_dict)
equilibrium_phases.set_ic(1)
equilibrium_phases.data

# %% [markdown]
# ### KINETICS Block
#
# See PHREEQC3 Manual, page ??

# %%
#kinetics phases
kinetic_phases_df = pd.read_csv(kinetic_phases_filepath, comment = '#',)
kinetic_phases_df

# %%
kinetic_phases_dict = mf6rtm.utils.parse_kinetics_dataframe(kinetic_phases_df)
kinetic_phases = mf6rtm.mup3d.KineticPhases(kinetic_phases_dict)
kinetic_phases.set_ic(1)
kinetic_phases.data

# %% [markdown]
# ### SURFACE Block
#
# See PHREEQC3 Manual, page ??

# %%
#kinetics phases
surfaces_df = pd.read_csv(surfaces_filepath)
surfaces_df

# %%
surfaces_dict = mf6rtm.utils.surfaces_csv_to_dict(surfaces_filepath)
surfaces = mf6rtm.mup3d.Surfaces(surfaces_dict)
surfaces.set_ic(1)
surfaces.data

# %% [markdown]
# ### Create reaction model (RM) instance with `mf6rtm` `Mup3d` class

# %%
# create model class, with solution initial conditions
reaction_model = mf6rtm.mup3d.Mup3d(simulation_name, solutions, nlay, nrow, ncol)

# set model workspace for saving outputs
reaction_model.set_wd(sim_ws)

# set Phreeqc database
reaction_model.set_database(phreeqc_database_filepath)

reaction_model.set_initial_temp([7., 7., 7.])

# set chemistry domain initilization to model object
reaction_model.set_exchange_phases(exchanger)
# reaction_model.set_phases(equilibrium_phases)
# reaction_model.set_phases(surfaces)
# reaction_model.set_phases(kinetic_phases)

# set Phreeqc postfix file
reaction_model.set_postfix(postfix_filepath)

print(reaction_model.name, reaction_model.grid_shape)

# %% [markdown]
# ### Set Component H2O to transport excess H & O
#
# [`SetComponentH2O()`](https://usgs-coupled.github.io/phreeqcrm/namespacebmiphreeqcrm.html#a0e152e5b6933e3e6bd8c79245917639a) is a PhreeqcRM function to select whether to include H2O in the component list. 
# - The concentrations of H and O must be known accurately (8 to 10 significant digits) for the numerical method of PHREEQC to produce accurate pH and pe values. 
# - Because most of the H and O are in the water species, it may be more robust (require less accuracy in transport) to transport the excess H and O (the H and O not in water) and water. 
# - The default setting (true) is to include water, excess H, and excess O as components. 
# - A setting of false will include total H and total O as components. 
# - `SetComponentH2O` must be called before `FindComponents`. 
#
# NOTE: The default for `mf6rtm` is FALSE, to use total H & O as components.

# %%
reaction_model.set_componenth2o(True) # True = transport H20 and excess H & O

# %% [markdown]
# ### Initialize IC Chemistry over Model Grid 
# This creates a PhreeqcRM instance based on components in Solution Blocks assigned initial conditions over the grid. It then runs a PHREEQC time zero equilibrium calculation for inital speciation.

# %%
# Intializing the mup3d class calculates the equilibrated
# initial concentration array
# NOTE: It appears that nthreads cannot be increased above 1 if using the Python phreeqcrm package
# See https://github.com/p-ortega/mf6rtm/issues/54

reaction_model.initialize(
    nthreads=4, 
    add_charge_flag=True,
)
# NOTE: If this hangs, check all input files for "!" 

# %%
reaction_model.phreeqc_rm.GetThreadCount()

# %%
# Dictionary of solution concentrations in units of moles per m^3 (or mmol/L), 
# and structured to match the shape of Modflow's grid
reaction_model.sconc  # comment out to reduce outputs
reaction_model.sconc['Charge'][wel_lay,:,wel_cellnum]  # or use it to find value at a single cell


# %%
reaction_model.components

# %%
# 1D array of concentrations in units of mol/L 
# structured for PhreeqcRM `GetConcentrations()` and BMI with
# component concentratinon arrays for the grid ordered as `model.components`
# Equivalent to `c_dbl_vect` (concentration double vector) in mf6rtm source code
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

# %% [markdown]
# ### Initialize BC Chemistry for all Boundaries

# %%
# Set and initialize stress period chemical concentrations for each BC
reaction_model.set_chem_stress(wellchem)
reaction_model.set_chem_stress(chdchem)
reaction_model.set_chem_stress(ghbchem)

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
# Append Conc data to Well Stress Period Data list, 
# NOTE: This is done in the "Add Chem to Modflow: Add Chem Components to Stress Period Data" section below

# %% [markdown]
# ## Get Output Variable Names

# %%
reaction_model.phreeqc_rm.GetSelectedOutputHeadings()

# %% [markdown]
# ## Unit Conversions
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
reaction_model.sconc['Ca']

# %%
# create new gwt models for each component
porosity = 0.3
dispersivity = 0.00656  # ft = 0.002 # Longitudinal dispersivity (m)
gwf_name = "flow"

for component_name in component_name_l:
    print("Adding gwt model for: " + component_name)
    gwt_name = "trans-" + component_name
    sim = utils.create_mf6_gwt(
        sim, gwf_name, gwt_name, component_name, 
        reaction_model.sconc[component_name],
        porosity, dispersivity
    )


# %%
# Confirm Modflow models in the simulation
sim.model_names

# %%
# Confirm initial condition concs for Na, from `mup3d.sconc`
# units of moles per m^3 (or mmol/L), 
sim.get_model('trans-Ca').ic.strt.array

# %% [markdown]
# ## Add Chem Components to Stress Period Data

# %% [markdown]
# ### WEL Chemistry

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

# %% [markdown]
# ### CHD Chemistry

# %%
############################################################################
### Add new components to MF6 chd spd and auxvar
############################################################################

# units of moles per m^3 (or mmol/L), from mf6rtm reaction_model 
chd_conc = reaction_model.chd.data[0]

# load wel package and stress period data
chd = gwf.chd
spd_chd_dict = chd.stress_period_data.get_data(full_data=True) 

# modify chd spd data
new_chd_spd = {}
for kper, records in spd_chd_dict.items():
    updated_record = utils.modify_chd_spd(records, component_name_l, chd_conc)
    new_chd_spd[kper] = np.rec.array(updated_record)

# set new aux variables
chd_spd_dtype = list(new_chd_spd[0].dtype.names)
new_chd_auxvar = chd_spd_dtype[2:-1]  # "2:-1" --> excludes wel parameters from auxvars
chd.auxiliary = new_chd_auxvar

# set stress period data to new_wel_spd that includes added components
chd.stress_period_data.set_data(new_chd_spd)


# %%
# Confirm values are set as expected
display(ic_df.T)
chd.stress_period_data.get_dataframe()[0].loc[0:1]

# %% [markdown]
# ### GHB Chemistry

# %%
############################################################################
### Add new components to MF6 ghb spd and auxvar
############################################################################

# units of moles per m^3 (or mmol/L), from mf6rtm reaction_model 
ghb_conc = reaction_model.ghb.data[0]

# load wel package and stress period data
ghb = gwf.ghb
spd_ghb_dict = ghb.stress_period_data.get_data(full_data=True) 

# modify wel spd data
new_ghb_spd = {}
for kper, records in spd_ghb_dict.items():
    updated_record = utils.modify_ghb_spd(records, component_name_l, ghb_conc)
    new_ghb_spd[kper] = np.rec.array(updated_record)

# set new aux variables
ghb_spd_dtype = list(new_ghb_spd[0].dtype.names)
new_ghb_auxvar = ghb_spd_dtype[3:-1]  # "2:-1" --> excludes wel parameters from auxvars
ghb.auxiliary = new_ghb_auxvar

# set stress period data to new_wel_spd that includes added components
ghb.stress_period_data.set_data(new_ghb_spd)

# %%
# Confirm values are set as expected
display(ic_df.T)
ghb.stress_period_data.get_dataframe()[0].loc[0:1]

# %% [markdown]
# ## Modify timestep length

# %%
# Modify number of timesteps per stress period
# modify tdis to change timestep length and total simulation time
change_nstp = True
    # if False, timestep lenght varies by stess period
nstp_multiplier = 8
number_of_days_last_stressperiod = 200.

if change_nstp == True:
    tdis_spd = sim.get_package("tdis").perioddata.get_data(full_data=True)
    tdis_spd["nstp"] = tdis_spd["nstp"] * nstp_multiplier
    # Modify length of last stress period
    tdis_spd['perlen'][-1] = number_of_days_last_stressperiod
    tdis_spd['nstp'][-1] = int(number_of_days_last_stressperiod * nstp_multiplier / 20)
                    # dividing by 20 gives about double the length of other periods
    sim.get_package("tdis").perioddata.set_data(tdis_spd)

# %%
tdis_spd_df = pd.DataFrame.from_records(tdis.perioddata.get_data())
# Add step length column
tdis_spd_df['stp_len'] = tdis_spd_df['perlen'] / tdis_spd_df['nstp']
# Cumulative Days column
tdis_spd_df['cumulative_days'] =  tdis_spd_df['perlen'].cumsum()
# Add flow rates
tdis_spd_df['q'] = spd_wel_df['q']
# Calculate cumulative volume 
tdis_spd_df['cum_q-days'] = (tdis_spd_df['q'] * tdis_spd_df['perlen']).cumsum()
tdis_spd_df

# %% [markdown]
# ## Remove transport models that are not needed

# %%
# Remove transport models for testing
sim.remove_model('trans-tds')
sim.remove_model('trans-temp')

# %%
gwt_model_names = [name for name in sim.model_names 
                    if (sim.get_model(name).model_type == 'gwt6')]
print("Number of transport models: ",len(gwt_model_names))
gwt_model_names

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
# When just running MF6, before any coupling.
#
# For addtional result plots of the simple ASR Modflow 6 simulation used throughout this repository, see `sims/sim00-mf6only/mf6_explore.ipynb`

# %% [markdown]
# # Reactive Transport Simulation
# Using MF6RTM

# %%
# Does this cell run?
"Yes"

# %%
# Run the model using this wrapper function for `mf6rtm.solve(model.wd)`
reaction_model.run(nthread=4, min_concentration=0.0)

# %% [markdown]
# Using MF6RTM + Coal Ash Chemistry
# Add Arsenic!!!
# - Crashes at sp5-ts36. step_multiplier=12; Knobs: step_size=10, pe_step_size=5, diag_scale=false
# Reboot
# - Crashes at sp3-ts40?.nstep_multiplier=20; Knobs above & tolerance=1e-18
# - Crashes at sp6-ts60. step_multiplier=20; Knobs above & tolerance=1e-17
# Reboot
# - 3.5365 mins. No As. 4xnstp. +tolerance=1e-16
# - Crashes sp4-ts12. 4xnstp +tolerance=1e-18
# - 7.8593 mins. +O2. 8xnstp
# - Crashes sp2-ts16. 8xnstp +O2 +As
# - 6.45 mins. 8xnstp. +As -eqphase -surface= -kinetics

# %%
nstp_multiplier

# %% [markdown]
# ## Visualize MF6RTM Results from PhreeqcRM

# %%
# read in phreeqc selected output  
sout_df = pd.read_csv(
    sim_ws / 'sout.csv', 
    sep = ',', 
    skipinitialspace=True, 
    index_col="time",
)
# Move "cell" to second column
# 1. Pop the column to move it out of the DataFrame
column_to_move = sout_df.pop('cell')
# 2. Insert the column at index 0 (the front)
sout_df.insert(0, 'cell', column_to_move)
sout_df.info()
sout_df

# %%
sout_df.describe()

# %%
# Get timestamp for plots
formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
formatted_time

# %% [markdown]
# ## Holoviz Plots

# %%
import hvplot.pandas
import holoviews as hv

# %%
# wel_cellid
wel_cellid = (2, 462)
# wel_cellid = (1, 941) # biggest residual on gwf from mfsim.lst

# %%
# Plot one cell away from well
cell_to_plot = (wel_cellid[0], wel_cellid[1]+1)
cell_to_plot

# %%
# cell_flatid = cellid_flatmap[*cell_to_plot]
cell_flatid = 2376
cell_flatid

# %%
plot_df = sout_df.loc[sout_df.cell == cell_flatid]
plot_df.columns

# %%
arsenic_on = True if 'As(+5)(mol/kgw)' in plot_df.columns else False
arsenic_on

# %%
major_elements = ['Ca(mol/kgw)', 'Mg(mol/kgw)', 'Cl(mol/kgw)',
       'C(4)(mol/kgw)', 'Alk(eq/kgw)', 
       'm_MgX2(mol/kgw)', 'm_CaX2(mol/kgw)',]
majors_plot = plot_df[major_elements].hvplot(ylabel='Majors Conc (mol/kgw)', logy=True)

# %%
minor_elements = ['S(6)(mol/kgw)', 'S(-2)(mol/kgw)', 'Fe(2)(mol/kgw)', 'Fe(3)(mol/kgw)',
    'm_FeX2(mol/kgw)',
    'As(+5)(mol/kgw)' if arsenic_on else "",
    'O(0)(mol/kgw)',
]
minors_plot = plot_df[minor_elements].hvplot(
    ylabel='Minors Conc (mol/kgw)', 
    logy=True, ylim=(1e-24,5e-5)
)

# %%
ph_plot = plot_df[['pH']].hvplot(ylabel='pH')
pe_plot = plot_df[['pe']].hvplot(ylabel='pe')

# %%
charge_balance = ['pct_err', ] # 'charge(eq)'
charge_plot = plot_df[charge_balance].hvplot.line(ylabel='Percent error, 100*(Cat-|An|)/(Cat+|An|)', logy=False)

# %%
plot_list = [majors_plot, minors_plot, ph_plot, pe_plot, charge_plot]
chem_layout_plot = hv.Layout(plot_list).cols(1).opts(
    title=f'MF6RTM Results for "{simulation_name}" at cellid {cell_to_plot}, run {formatted_time}',
    shared_axes=False, 
    axiswise=True,
)
chem_layout_plot

# %% [markdown]
# ## Save Figures & Outputs

# %%
plot_path = working_dir / f"{sim_ws}_plots"
plot_path.mkdir(exist_ok=True, parents=True)
plot_path

# %%
file_time = formatted_time.replace("-", ".").replace(":",".").replace(" ", "_")
plot_filename = (f'{simulation_name}_run_{file_time}').replace(" ", "_")
plot_filename

# %%
hv.save(chem_layout_plot, plot_path / f'{plot_filename}.png')
hv.save(chem_layout_plot, plot_path / f'{plot_filename}.html')

# %%
sout_df.to_parquet(plot_path / f'sout_df_{file_time}.parquet', compression='zstd', index=True)

# %%
nstp_multiplier = 8

# %%
shutil.copy2(sim_ws / "_phreeqc.chem.txt", plot_path / f"phreeqc.chem_{file_time}.txt")
shutil.copy2(sim_ws / "phinp.dat", plot_path / f"phinp_{file_time}_{nstp_multiplier}xnstp.dat")

# %% [markdown]
# # END

# %%
