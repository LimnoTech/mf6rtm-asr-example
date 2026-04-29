# ---
# jupyter:
#   jupytext:
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
# # Getting Started with this Repo
#
# This notebook helps you get started with using this MF6RTM Aquifer Storage Recovery (ASR) Example repository.
#
# This repo uses the conda for:
# - Virtural environment managment
# - Package managment for dependances
# - Setting up the mf6rtm repo in "develop" mode.
#

# %% [markdown]
# # Installation and Setup
#
# Create a custom conda virtual environment can be created using the `environment.yml` file included in this repo.
#
# See the repo's [`README.md`](https://github.com/LimnoTech/mf6rtm-asr-example/blob/main/README.md#install-development-environment-with-conda) file for detailed instructions.

# %% [markdown]
# ## Python Imports

# %%
import os
from pathlib import Path
from importlib.metadata import version

import flopy
from modflowapi import ModflowApi
import phreeqcrm

# %%
# Import the MFRTM package, installed using `conda develop`
import mf6rtm
version("mf6rtm")

# %% [markdown]
# NOTE: the notebook below runs from the `externalio` branch the upstream repo 
# (Pablo's) https://github.com/p-ortega/mf6rtm/tree/feature/externalio

# %%
from mf6rtm_asr_example import utils # from this repo

# %% [markdown]
# ### If you get `ModuleNotFoundError`
#
# ... you need to install this package into your environment using 
# [`conda develop`](https://docs.conda.io/projects/conda-build/en/latest/resources/commands/conda-develop.html) 
# command in your terminal with your local absolute path to the source directory
#  of this repo. Then restart the kernel.

# %% [markdown]
# ## Install `mf6rtm` using `conda develop`

# %%
# Find your current working directory, which should be folder for this notebook.
working_dir = Path.cwd()
# Find repository path (i.e. the parent to `/examples` directory for this notebook)
repo_path = working_dir
repo_path

# %%
# If MF6RTM is installed in the same directory as this repository,
# then `mf6rtm_source_path` should be the source directory
mf6rtm_source_path = repo_path.parent / "mf6rtm"

print(
    "MF6RTM Source path exists?",
    mf6rtm_source_path.exists(),
    mf6rtm_source_path,
)

assert mf6rtm_source_path.exists(), "Find `mf6rtm` source path"

# %% [markdown]
# Use the Jupyter [`!` shell command](https://jakevdp.github.io/PythonDataScienceHandbook/01.05-ipython-and-shell-commands.html) to run the `conda develop {source_path}` terminal command directly from this notebook.
#
# NOTE: The Jupyter [`%conda` magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-conda) will note work with `%conda develop {source_path}` in Windows, because it prepends the local working directory to the source path, inserting the wrong path to the `conda.pth` file.

# %%
# !conda develop {mf6rtm_source_path}

# %%
# Also add this repository's source path for any local modules
# !conda develop {repo_path / 'src'}

# %% [markdown]
# If the path was added, Restart the kernel and rerun the cells above.
#
# NOTE: if you have already done this once before, you may need to manually 
# delete previous paths and add the path to:
# `~/miniconda3/envs/modflow/lib/python3.12/site-packages/conda.pth`

# %% [markdown]
# ## Alternative: add `mf6rtm` path directly to `conda.pth` file
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
    print(f'The `conda.pth` file exists at {conda_pth_filePath}\n')
    print('It includes these contents')
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
# # Confirm Paths to MF6 Executable & Library
# Different versions can be downloaded from: https://github.com/MODFLOW-ORG/executables to a folder similar to this: `bin/mf6.5.0/macarm` 
#
# On Mac, will need to give permissions with these terminal commands from the 
# ```sh
# xattr -dr com.apple.quarantine mf6
# xattr -dr com.apple.quarantine libmf6.dylib
# ```
#

# %%
# Check Modflow 6 version installed with ModflowAPI
try:
    mf6_exe = "mf6"
    dll = "libmf6"
    mf6_version = !{mf6_exe} --version
    mf6dll_version = ModflowApi("libmf6").get_version()
    print(f"Executable & library installed with modflowapi: {mf6_version[1]}, dll: {mf6dll_version}")
except Exception:
    print("Modflow executables not found in environment")

# %%
# Select option to use alternate versions
use_version_installed_with_modflowapi = False

# user = "Laren"
user = "Anthony"

# version = "6.4.2"
version = "6.5.0"

os = "macarm"


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

# %% [markdown]
# # END
# You are ready to use the other notebooks in this repo.
