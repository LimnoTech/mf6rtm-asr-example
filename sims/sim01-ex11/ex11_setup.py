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
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: modflow
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ASR Test Simulation: Phreeqc Example 11 Chemistry
#
# NOTE: This [Jupytext](https://jupytext.readthedocs.io/en/latest/index.html) paired notebook, with paired `.py` and `.ipynb` files. 
# - If using VS Code, install the the [Jupytext Sync extension](https://jupytext.readthedocs.io/en/latest/vs-code.html) for maximum benefit.
#
#
# The workflow for this example:
# - Read geochemical components and their initial and boundary concentrations from PHREEQC input files
# - Create new Modflow 6 transport model for each aqueous phase (components in the Solution blocks) and add their initial concentrations over the entire DISV grid.
# - Modify the Modflow 6 Flow Well package Stress Period Data (SPD) by adding Solution component concentrations.
# - Run the modified Modflow 6 for conservative transport of all components (i.e. no coupling to PHREEQC)
# - Run the coupled Modflow 6 & PHREEQC models for the entire simulation
#
#
#
# ## Simple ASR Test Case
#
# Grid type: DISV  
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
# # %matplotlib widget

import flopy
from modflowapi import ModflowApi
# import phreeqcrm

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
# NOTE: the notebook below runs from the `externalio` branch the upstream repo 
# (Pablo's) https://github.com/p-ortega/mf6rtm/tree/feature/externalio

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
sim_ws = working_dir / 'ws'
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
mf6_inputs_path = repo_path / 'data' / 'MF6_ASR_DISV_inputs'
mf6_input_files = os.listdir(mf6_inputs_path)

# %%
# Copy input files to simulation workspace directory)
for file in mf6_input_files:
    shutil.copy2(mf6_inputs_path / file, sim_ws)

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
assert sim_nam_file_path.exists(), "MF6 sim file exists!"
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
print("PHREEQC input files exist?", solutions_filepath.exists(), exchanges_filepath.exists(), )

# %%
# Path to file with PHREEQC Input "postfix" instructions
# to be appended to the PHREEQC Input file (*.pqi) created by mf6rtm
postfix_filepath = sim_ws /  'chem_postfix.phqr'
postfix_filepath.exists()

# %%
# Select PHREEQC database file
phreeqc_database_file = "phreeqc.dat"
phreeqc_databases_path = repo_path / "data" / "chem_databases"
phreeqc_database_filepath = phreeqc_databases_path / phreeqc_database_file
print("PHREEQC database file exists?", phreeqc_database_filepath.exists())

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
# Copy executable and library to simulation workspace
shutil.copy2(mf6_exe, sim_ws)
shutil.copy2(dll, sim_ws)
(sim_ws/mf6_exe.name).exists()

# %% [markdown]
# ## Utility Functions
#
# These functions are available in `src/utils.py`:
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

# %%
# TODO: Calculate volume of cells near well screen
cells_df.loc[494:496]

# %%
# Get Cell Index to (cellid_layer, cellid_cell) mapping

# %%
nxyz/nlay

# %%
np.tile(np.arange(0,nlay), nxyz/nlay)

# %%
pd.DataFrame(
    index=np.arange(0,nxyz),
    columns=[
        np.arange(0,nlay),
        np.arange(0,ncpl),
        
    ]
)

# %%
np.arange(0,nxyz)

# %% [markdown]
# ### Grid Cell Map

# %%
# Turn on interactive
# %matplotlib widget

# %%
# plot map view of grid showing order of grid cell ids and vertices from:
# https://modflow6-examples.readthedocs.io/en/latest/_notebooks/ex-gwf-u1disv.html
fig = plt.figure(figsize=(6,6))
fig.tight_layout()
ax = fig.add_subplot(1, 1, 1, aspect="equal")
pmv = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=0)
pmv.plot_grid()
pmv.plot_bc(name="ghb", alpha=0.75)
pmv.plot_bc(name="wel", alpha=0.75)
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")
for i, (x, y) in enumerate(
    zip(gwf.modelgrid.xcellcenters, gwf.modelgrid.ycellcenters)
):
    ax.text(
        x,
        y,
        f"{i + 1}",
        fontsize=6,
        horizontalalignment="center",
        verticalalignment="center",
    )
v = gwf.disv.vertices.array
ax.plot(v["xv"], v["yv"], "yo")
for i in range(v.shape[0]):
    x, y = v["xv"][i], v["yv"][i]
    ax.text(
        x,
        y,
        f"{i + 1}",
        fontsize=5,
        color="red",
        horizontalalignment="center",
        verticalalignment="center",
    )
plt.show()

# %% [markdown]
# #### Screenshot of Grid Cell Map
# ![image.png](attachment:image.png)

# %%
# Turn off interactive widget
# %matplotlib inline

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
# modify well q to try and address convergence issues
for sp in range(len(spd_wel_dict)):
    spd_wel_dict[sp]['q'] = spd_wel_dict[sp]['q'] / 2.

# reset wel stress period data using modified q
wel.stress_period_data = spd_wel_dict
spd_wel_dict_edit = wel.stress_period_data.get_data(full_data=True)

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
ncomps_by_nxyz_conc_array

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
tdis_spd = sim.get_package("tdis").perioddata.get_data(full_data=True)
tdis_spd
tdis_spd["nstp"] = tdis_spd["perlen"]  # each timestep = 1.0 days
#tdis_spd["nstp"] = tdis_spd["perlen"]  # set number of steps (nstp) equal to stress period length (perlen) so dt = 1 day for each stress period
# tdis_spd['nstp'][0] = 20 # set first stress period to 20 days with 1 timestep per day
# tdis_spd['perlen'][0] = 20
tdis_spd["nstp"]
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
# Conc is a nested array of these shapes
display(conc.shape, conc[0].shape)

# %% [markdown]
# ### Head

# %%
# plot head
f = 101
fig = plt.figure(num=f, figsize=(18, 5))
plt.plot(times_h, head[:, wel_lay, 0, wel_cellnum], marker=".")
f = f + 1
fig.show()


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
# ### temp and tds gwt output

# %%
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
t_l = [0, 1, 10, 50, 200, 400, -1]  # list of timestep index (NOT actual time/days)
for t in t_l:
    fig = plt.figure(figsize=(24, 4))
    ax = fig.add_subplot(1, 1, 1, aspect="auto")
    ax.set_title("conc of " + component_name_l[s] + " at timestep index t=" + str(t))
    mapview = flopy.plot.PlotMapView(gwf, layer=2)  # ,extent=(0,0.08,0,1.))
    patch_collection = mapview.plot_array(conc[s][t, :, :, :])  # ,vmin=0., vmax=0.2)
    linecollection = mapview.plot_grid()
    cb = plt.colorbar(patch_collection, shrink=0.75)

# %%
# Cause BREAK
# assert 1 != 1

# %% [markdown]
# # Reactive Transport Simulation
# Using MF6RTM

# %%
# Run the model using this wrapper function for `mf6rtm.solve(model.wd)`
reaction_model.run()

# %% [markdown]
# ## Visualize MF6RTM Results

# %%
sout_df = pd.read_csv(
    sim_ws / 'sout.csv', 
    sep = ',', 
    skipinitialspace=True, 
    index_col=[0],
)
sout_df.info()
sout_df.head()

# %%
sout_df.cell

# %%
cell_num = 511
sout_df[(sout_df.cell == cell_num)].plot(y=['Na', 'K'], logy=False)

# %% [markdown]
# # END

# %%
