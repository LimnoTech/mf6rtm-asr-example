# MF6RTM Example for Aquifer Storage Recovery (ASR)

An example for the [MF6RTM (Modflow 6 Reactive Transport Model) package](https://github.com/p-ortega/mf6rtm) for a simple Aquifer Storage and Recovery (ASR) use case over a 3D grid (DISV).

This repository provides a single Modflow 6 simulation for a 3D unstructured layered grid (5 layers each with 1032 cells) specified with the Discretization by Vertices (DISV) Package centered on a single ASR well specified with the WEL package.
This simulation serves as a unified foundation to develop and test:

- A range different geochemical scenarios
- Reproducible workflows for setting up and simulating scenarios
- Utilites to facilitate workflows
- Improvements to the [`mf6rtm`](https://github.com/p-ortega/mf6rtm) package


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
