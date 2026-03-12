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
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: modflow
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ASR Simulation 2: Grid 2 (10ft near well) with Chemistry from Phreeqc Example 11 
#
# This simulation adds **cation exhange reaction chemistry** -- from the PHREEQC-3 Manual's Example 11 (Parkhurst and Appelo 2013) -- to the 3D transport models of the simple ASR test case.
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
import utils # from this repo

# %% [markdown]
# ### If you get `ModuleNotFoundError`
#
# Run the `01-GettingStarted.ipynb` notebook to install `mf6rtm` using `conda develop`.

# %% [markdown]
# ## Set Paths to Input and Output Files with `pathlib`
#
# Use the [pathlib](https://docs.python.org/3/library/pathlib.html) library 
# (built-in to Python 3) to manage paths indpendentely of OS or environment. 
# See this [blog post]
# (https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f) 
# to learn about the many benefits over using the `os` library.

# %%
# Find your current working directory, which should be folder for this notebook.
working_dir = Path.cwd()
# Find repository path (i.e. the parent to `/examples` directory for this notebook)
repo_path = working_dir.parent.parent
repo_path

# %%
simulation_name = working_dir.name
simulation_name

# %%
# Path to simulation workspace, which is git-ignored and 
# will get over-written with each run of this notebook
sim_ws = working_dir / 'ws2' # Grid 2
sim_ws.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### Reset Workspace

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
chem_inputs_path = working_dir / "chem_inputs"
chem_input_files = os.listdir(chem_inputs_path)
chem_input_files

# %%
# Copy input files to simulation workspace directory (i.e. project path)
for file in chem_input_files:
    shutil.copy2(chem_inputs_path / file, sim_ws)

# %%
# Path to PHREEQC Block Input CSV Files
solutions_filepath = sim_ws / "chem_solutions.csv"
exchanges_filepath = sim_ws / "chem_exchanges.csv"
assert solutions_filepath.exists() and exchanges_filepath.exists()

# %%
# Path to file with PHREEQC Input "postfix" instructions
# to be appended to the PHREEQC Input file (*.pqi) created by mf6rtm
postfix_filepath = sim_ws /  'chem_postfix.phqr'
assert postfix_filepath.exists()

# %%
# Select PHREEQC database file
# phreeqc_database_file = "phreeqc.dat" # used in Ex6?
phreeqc_database_file = 'pht3d_datab.dat' # used in Ex4
phreeqc_databases_path = repo_path / "data" / "chem_databases"
phreeqc_database_filepath = phreeqc_databases_path / phreeqc_database_file
assert phreeqc_database_filepath.exists(), "PHREEQC database file missing"

# %%
# Paths to PHREEQC configuration files that will be created by mf6trm
# PHREEQC Input file (*.pqi)
phreeqc_input_filepath = sim_ws / "phinp.dat"
# PhreeqcRM YAML config file
phreeqcrm_yaml_filepath = sim_ws / "mf6rtm.yaml"
print("MF6RTM-created PHREEQCRM YAML file exists?", phreeqcrm_yaml_filepath.exists())

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
os = "macarm"

# version = "6.4.2"
version = "6.5.0"
# version = "6.7.0.dev3"

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
        mf6_bin_path = Path(r"C:/program files/gms 10.8 64-bit/python/lib/site-packages/xms/executables/modflow6")
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

# %%
# Copy executable and library to simulation workspace
shutil.copy2(mf6_exe, sim_ws)
shutil.copy2(dll, sim_ws)
(sim_ws/mf6_exe.name).exists()

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
# volume of cells near well screen
cell_volumes[2, wel_cellnum-6:wel_cellnum+6]

# %%
# TODO: Get flat Cell Index to (cellid_layer, cellid_cell) mapping
# for exploring phreeqcrm outputs

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
display(spd_wel_dict)

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

reaction_model.initialize(add_charge_flag=True)

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

# %%
# Dictionary of concentrations in units of moles per m^3 (or mmol/L), 
# and structured to match the shape of Modflow's grid
reaction_model.sconc

# %% [markdown]
# #### Aside to test approaches for reshaping

# %%
# create alias for testing current implementation
# `c_dbl_vect` is the concentration double vector in units of mol/L
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
change_nstp = True
if change_nstp == True:
    tdis_spd = sim.get_package("tdis").perioddata.get_data(full_data=True)
    #tdis_spd = tdis_spd[0:5]
    #tdis_spd["nstp"] = tdis_spd["perlen"]   # each timestep = 1 day
    #tdis_spd["nstp"] = tdis_spd["perlen"]  # set number of steps (nstp) equal to stress period length (perlen) so dt = 1 day for each stress period
    #for t in range(len(tdis_spd)):
    #    tdis_spd['nstp'][t] = 1
    #    tdis_spd['perlen'][t] = 1.
    #tdis_spd['nstp'][0] = 20 # set first stress period to 20 days with 1 timestep per day
    # tdis_spd['perlen'][0] = 20
    tdis_spd['perlen'][-1] = 600.
    tdis_spd['nstp'][-1] = 60
    sim.get_package("tdis").perioddata.set_data(tdis_spd)

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
# Conc is a nested array of these shapes
display(conc.shape, conc[0].shape)

# %% [markdown]
# ### Conc Timeseries near Well 

# %%
# Create list of components to plot based on intersection with transported components
components_to_plot = [c for c in component_name_l if c in ['Ca', 'Cl', 'K', 'N', 'Na']]
components_to_plot

# %%
k = wel_lay  # layer index
cnum = wel_cellnum  # cell number
for c in range(len(component_name_l)):
    if component_name_l[c] in components_to_plot:
        fig = plt.figure(num=101, figsize=(10, 5))
        plt.plot(times_c[c], conc[c][:, k, 0, cnum], label=component_name_l[c])
        plt.title("[" + str(k) + "," + str(cnum) + "]")
        plt.legend()

# %%
# Get Concentration Values
c = -1
time = 5
layer = 2
cell_num = wel_cellnum + 20
print(component_name_l[c], cell_num)
conc[c][0:time, layer, 0, cell_num]

# %% [markdown]
# ### Conc Cross Sections

# %%
# xsection
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

# %%
# creates a cross section along the line specified above for each timestep in t_l
s = 4  # solute index for Cl
t_l = [1, 10, 25, 50, -1]  # list of timestep index (NOT actual time/days)
normalize = True
if normalize == True:
    scale = 50
else:
    scale = 100
for t in t_l:
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(
        spdis[t], gwf, head=head[t]
    )
    fig = plt.figure(figsize=(9, 2.5))
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

# %% [markdown]
# ### Conc Map View

# %%
s = 3  # solute index for Ca
t_l = [0, 1, 10, 50, -1]  # list of timestep index (NOT actual time/days)
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
#
# Run times:
# - 10% faster than grid 1

# %% [markdown]
# ## Visualize MF6RTM Results

# %%
# read in mf6 conc results
sim_rxn = flopy.mf6.MFSimulation.load(
    sim_ws=sim_ws,
    exe_name=mf6_exe,  #'mf6',
    verbosity_level=0,
)
conc_rxn = utils.get_concentrations(sim_rxn, component_name_l)
times_c_rxn = utils.get_times_c(sim_rxn, component_name_l)

# %%
# read in phreeqc results 
sout_df = pd.read_csv(
    sim_ws / 'sout.csv', 
    sep = ',', 
    skipinitialspace=True, 
    index_col=[0],
)
sout_df.info()
sout_df

# %%
component_name_l

# %%
list_offset = len(component_name_l) - len(components_to_plot)
list_offset

# %%
# plot mf6 transport only, mf6 conc with rxn, and phreeqc
k = 2 #wel_lay  # layer index
cnum = wel_cellnum  # cell number
colors_c     = ['paleturquoise','plum','gold','darkseagreen','cornflowerblue'] 
colors_c_rxn = ['teal','purple','darkgoldenrod','darkgreen','midnightblue']
f = 101 # Figure Number
for c in range(len(component_name_l)):
    if component_name_l[c] in components_to_plot:
        # print(component_name_l[c])
        fig = plt.figure(num=f, figsize=(9, 5))
        plt.plot(times_c[c], conc[c][:, k, 0, cnum], color=colors_c[c-list_offset],label=component_name_l[c],marker='.',markersize=12)
        plt.plot(times_c_rxn[c], conc_rxn[c][:, k, 0, cnum], color=colors_c_rxn[c-list_offset], label=component_name_l[c]+'_rxn',marker='.')
        plt.title("Grid Cell ID: [" + str(k) + "," + str(cnum) + "]")
        leg = plt.legend(loc='center right', bbox_to_anchor=(1.17, 0.5))
plt.xlabel('Time (d)')
plt.ylabel('Concentration (mmol/kg)')
plt.tight_layout()
plt.show()

# %%
component_name_l

# %%

################################################################################
### Plot cross section (xsect) of concentration of MF6RTM with rxn results
################################################################################

# OPTIONS # 
plot_bcs = False # if True, plots boundary conditions for wel and chd package
normalize = True # if True, specific discharge is normalized and only shows direction
plot_spdis = False # if True, plots specific discharge on cross section
# component_name_l = ['H', 'O', 'Charge', 'Ca', 'Cl', 'K', 'N', 'Na']
s = component_name_l.index("Ca")  # solute index for Ca
t_l = [0, 1,10,20, 30, 40, 50, 60, 80, -1]  # list of timestep index (NOT actual time/days)
# OPTIONS #

# to plot a cross section with disv, you have to make a line to plot along
line = np.array([(694298, 1025435), (6999092, 1025435)]) # goes through wel cellID 496
# creates a plot showing where the line is on the grid to make the cross section plot
fig = plt.figure(num=f,figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1, aspect="auto")
ax.set_title("Vertex Model Grid (DISV) with cross sectional line")
# use PlotMapView to plot a DISV (vertex) model
mapview = flopy.plot.PlotMapView(gwf, layer=1)  # ,extent=(0,0.08,0,1.))
if plot_bcs == True:
    mapview.plot_bc("WEL-1")
    mapview.plot_bc("CHD-1")
linecollection = mapview.plot_grid()
# plot the line over the model grid
lc = plt.plot(line.T[0], line.T[1], "r--", lw=0.8)
plt.show()
f = f + 1

# creates a cross section along the line specified above for each timestep in t_l
# reads concentration data from modflow6 output files for MF6RTM simulation
sim = flopy.mf6.MFSimulation.load(
    sim_ws=sim_ws,
    exe_name=mf6_exe,
    verbosity_level=0,
)
conc = utils.get_concentrations(sim, component_name_l)
times_c = utils.get_times_c(sim, component_name_l)

if normalize == True:
    scale = 50
else:
    scale = 100
for t in t_l:
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(
        spdis[t], gwf, head=head[t]
    )
    fig = plt.figure(num = f,figsize=(6, 5))
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
    patch_collection = xsect.plot_array(conc[s][t, :, :, :], vmin=0.0, vmax=1.2)
    line_collection = xsect.plot_grid(linewidth=0.5)
    if plot_spdis == True:
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
    plt.xlabel('distance (m)')
    plt.ylabel('elevation (m)')
    plt.tight_layout()
    plt.show()
    f = f + 1

# %% [markdown]
# # END

# %%
