# MF6RTM Example for Aquifer Storage Recovery (ASR)

An example for the [MF6RTM (Modflow 6 Reactive Transport Model) package](https://github.com/p-ortega/mf6rtm) for a simple Aquifer Storage and Recovery (ASR) use case over a 3D grid (DISV).

This repository provides a single Modflow 6 simulation for a 3D unstructured layered grid (5 layers each with 1032 cells) specified with the Discretization by Vertices (DISV) Package centered on a single ASR well specified with the WEL package.
This simulation serves as a unified foundation to develop and test:

- A range different geochemical scenarios
- Reproducible workflows for setting up and simulating scenarios
- Utilites to facilitate workflows
- Improvements to the [`mf6rtm`](https://github.com/p-ortega/mf6rtm) package


## Jupyter Notebooks paired with Jupytext

Many Jupyter Notebooks are automatically paired with `.py` files via [Jupytext](https://jupytext.readthedocs.io/en/latest/index.html). Editing one file should automatically sync code and markdown to the other file with every open or save.

If using VS Code, install the the [Jupytext Sync extension](https://jupytext.readthedocs.io/en/latest/vs-code.html) for maximum benefit.


## Installation

We recommend using [pixi](https://pixi.prefix.dev/latest/), the next-generation reproducible package management tool built on [conda](https://docs.conda.io/projects/conda/en/stable/) tooling. 

A major benefit is that Pixi itself can be installed on any platform, including linux supercomputers, with a shell script and without needing a pre-existing Python environment.

If you are new to pixi but familiar with conda, this [Switching from Conda](https://pixi.prefix.dev/latest/switching_from/conda/) documentation succinctly compares similarities and differences.

Alternately, use a conda environment with the same dependencies.

### Install Development Environment with Pixi (Recommended)

### 1. Install Pixi

Follow [Pixi Installation](https://pixi.prefix.dev/latest/installation/) instructions for your platform.

### 2. Clone or Download this Repository

From this Github page, click on the green "Code" dropdown button near the upper right. Select to either "Open in GitHub Desktop" (i.e. git clone) or "Download ZIP". We recommend using GitHub Desktop, to most easily receive updates, stage commits, and resolve merge conficts.

Place your copy of this repo in any convenient location on your computer.

### 3. Create a project-specific workspace using pixi

Create a project-specific environment and workspace from the `pixi.toml` or `pyproject.toml` manifest file. 

From your terminal or console, navigate to the directory of the repository you just cloned. To install the development environment, execute the following command: 

```sh
pixi install
```

To activate the newly created environment, execute the following command:
 
```sh
pixi shell
```

Note that VSCode does not always detect your new pixi environments, so you may need to **Manually Select the Interpreter**: 

- Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`).
- Type and select "Python: Select Interpreter".
- Select "Enter interpreter path..." and then "Find...".
- Navigate to your project's pixi environment directory. The default path is usually within your project folder at `.pixi/envs/default/bin/` python (or Scripts\python.exe on Windows).
- Select the `python` or `python.exe` executable.

Installing the [Pixi Code](https://marketplace.visualstudio.com/items?itemName=renan-r-santos.pixi-code) VSCode extension might also help.


### Install Development Environment with Conda

Follow these steps to install using the [conda](https://docs.conda.io/en/latest/) package manager.

#### 1. Install Miniconda or Anaconda Distribution

We recommend installing the light-weight [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) that includes Python, the [conda](https://conda.io/docs/) environment and package management system, and their dependencies.

If you have already installed the [**Anaconda Distribution**](https://www.anaconda.com/download), you can use it to complete the next steps, but you may need to [update to the latest version](https://docs.anaconda.com/free/anaconda/install/update-version/).

If you are on Windows, we recommend initializing conda for all your command prompt terminals, by opening the "Anaconda Prompt" console and typing this command:

```shell
conda init --all
```

#### 2. Clone or Download this Repository

From this Github page, click on the green "Code" dropdown button near the upper right. Select to either "Open in GitHub Desktop" (i.e. git clone) or "Download ZIP". We recommend using GitHub Desktop, to most easily receive updates.

Place your copy of this repo in any convenient location on your computer.

#### 3. Create a Conda Environment for this Repository

We recommend creating a custom virtual environment with the same software dependencies that we've used in development and testing, as listed in the [`environment.yml`](environment.yml) file. 

Create a project-specific environment using this [conda](https://conda.io/docs/) command in your terminal or Anaconda Prompt console. If necessary, replace `environment.yml` with the full file pathway to the `environment.yml` file in the local cloned repository.

```shell
conda env create --file environment.yml
```

Alternatively, use the faster [`libmamba` solver](https://conda.github.io/conda-libmamba-solver/getting-started/) with:

```shell
conda env create -f environment.yml 
```

Activate the environment using the instructions printed by conda after the environment is created successfully.

To update your environment run the following command:  

```shell
conda env update --file environment.yml --prune 
```
