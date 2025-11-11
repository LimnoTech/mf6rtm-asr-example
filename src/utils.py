"""ASR Reactive Transport Utilities
To complement functions in `flopy` and `mf6rtm`
"""

from pathlib import Path
import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
import flopy

molecular_weight_dict = {
    "H": 1.0079,
    "O": 15.9994,
    "Charge": 5.4858e-4,  # electron mass in g/mol
    "Ca": 40.078,
    "Cl": 35.453,
    "K": 39.098,
    "N": 14.007,
    "Na": 22.990,
    "C": 12.0107,
    "P": 30.973762,
    "Si": 28.0855,
}
"""Dictionary of Molecular Weights, in g/mol, of geochemical 
components used by PHREEQC."""


# Function for running the Modflow 6 input files
def run_models(sim, silent=True):
    """runs modflow 6 simulation"""
    success, buff = sim.run_simulation(silent=silent)
    assert success, buff


def write_models(sim, silent=True):
    """writes modflow 6 input files"""
    sim.write_simulation(silent=silent)


# Functions for adding groundwater transport models
def create_mf6_gwt(
    sim: flopy.mf6.MFSimulation,
    gwf_name: str,
    gwt_name: str,
    component_name: str,
    ic_conc: float,
    porosity: float,
    dispersivity: float,
):
    """
    sim            : mf6 simulation (MFSimulation)
    gwf_name       : groundwater flow model name (str)
    gwt_name       : groundwater transport model (str)
    component_name : name of chemical species/component/solute (str)
    ic_conc        : intial conditions concentration for each grid cell (array)

    Note: Only adds auxvars to the WEL package. For DISP package assumes that
          transverse dispersivity is 0.1*longitudinal dispersivity. Output for
          each solute is written to gwt.name_output dir

    """

    gwf = sim.get_model(gwf_name)
    gwf_disv = gwf.get_package("DISV")

    # make output directory for .lst, .cbc, and .unc files
    output_dir_path = sim.sim_path / f"{gwt_name}_output"
    if output_dir_path.exists() == False:
        output_dir_path.mkdir(parents=True,exist_ok=True)

    gwtlst_file_path = sim.sim_path/ f"{gwt_name}_output" /"model.lst"
    gwt = flopy.mf6.ModflowGwt(sim, modelname=gwt_name,list=gwtlst_file_path)

    imsgwt = flopy.mf6.ModflowIms(
        sim,
        print_option="ALL",
        complexity="moderate",
        linear_acceleration="BICGSTAB",
        filename=f"{gwt.name}.ims",
    )

    sim.register_ims_package(imsgwt, [gwt.name])

    disvgwt = flopy.mf6.ModflowGwtdisv(
        gwt,
        length_units=gwf.disv.length_units.array,
        nlay=gwf.disv.nlay.array,
        ncpl=gwf.disv.ncpl.array,
        nvert=gwf.disv.nvert.array,
        top=gwf.disv.top.array,
        botm=gwf.disv.botm.array,
        vertices=gwf.disv.vertices.array,
        cell2d=gwf.disv.cell2d.array,
    )

    mst = flopy.mf6.ModflowGwtmst(gwt, porosity=porosity, first_order_decay=None)

    icgwt = flopy.mf6.ModflowGwtic(gwt, strt=ic_conc)

    adv = flopy.mf6.ModflowGwtadv(gwt, scheme="tvd")

    dsp = flopy.mf6.ModflowGwtdsp(
        gwt,
        xt3d_off=True,
        alh=dispersivity,
        ath1=dispersivity * 0.1,
        atv=dispersivity * 0.1,
    )

    # modify wel, chd, and ghb aux variables

    sourcerecarray = [(gwf.wel.package_name, "AUX", component_name)]  # ,
    # (gwf.chd.package_name, "AUX", "trans-"+component_name),
    # (gwf.ghb.package_name, "AUX", "trans-"+component_name)]
    ssm = flopy.mf6.ModflowGwtssm(gwt, sources=sourcerecarray)

    budget_file_path = sim.sim_path/ f"{gwt_name}_output" /f"{gwt_name}.cbc"
    conc_file_path = sim.sim_path/ f"{gwt_name}_output" /f"{gwt_name}.ucn"
    ocgwt = flopy.mf6.ModflowGwtoc(
        gwt,
        budget_filerecord=budget_file_path,
        concentration_filerecord=conc_file_path,
        concentrationprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("CONCENTRATION", "ALL"),("BUDGET", "ALL")],
    )

    gwfgwt = flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype="GWF6-GWT6",
        exgmnamea=gwf.name,
        exgmnameb=gwt.name,
        filename="gwfgwt_" + component_name + ".gwfgwt",
    )

    return sim


def get_times_c(sim, solute_names):
    """loads concentration for all solutes for full simulation lenght"""
    """ conc[0].shape = 
        (240, 2, 1, 80)
        ^     ^  ^  ^
        |     |  |  number of cells per layer (ncpl)
        |     |  dummy row dimension (always 1 for DISV)
        |     number of layers (nlay = 2)
        number of time steps (240)"""
    times_c = np.full((len(solute_names)), np.nan, dtype="object")
    for c in range(len(solute_names)):
        gwt_name = "trans-" + solute_names[c]
        gwt = sim.get_model(gwt_name)
        times_c[c] = gwt.output.concentration().get_times()

    return times_c


def get_concentrations(sim, solute_names):
    """loads concentration for all solutes for full simulation lenght"""
    """ conc[0].shape = 
        (240, 2, 1, 80)
        ^     ^  ^  ^
        |     |  |  number of cells per layer (ncpl)
        |     |  dummy row dimension (always 1 for DISV)
        |     number of layers (nlay = 2)
        number of time steps (240)"""
    conc = np.full((len(solute_names)), np.nan, dtype="object")
    for c in range(len(solute_names)):
        gwt_name = "trans-" + solute_names[c]
        gwt = sim.get_model(gwt_name)
        conc[c] = gwt.output.concentration().get_alldata()

    return conc


def convert_molL_gL(component_name, conc_molL):
    conc_l_gL = conc_molL * molecular_weight_dict[component_name]
    return conc_l_gL


def convert_molL_kgft3(component_name, conc_molL):
    conc_l_kgft3 = conc_molL * molecular_weight_dict[component_name] / 0.03531 / 1000
    return conc_l_kgft3


def modify_wel_spd(
    record: np.recarray,
    component_name_l: list,
    wel_conc: list,
):
    # get boundname and save for later
    boundname = list(record["boundname"])
    # remove tds, temp and boundname using rfn.drop_fields
    record_trimmed = rfn.drop_fields(record, ["boundname"])  #'tds','temp'

    # add wel conc data
    # numpy.lib.recfunctions.append_fields(base, names, data, dtypes=None, fill_value=-1, usemask=True, asrecarray=False)
    # base       = original recarray you want to add fields to
    # names      = string for one filed or list of strings for new field name(s)
    # data       = 1D list/array of values for the new field(s). For multiple
    #              fields use a list of arrays, where len(array) = len(base)
    # dtypes     = dtype (string or dtype obj) or list of dtypes for names
    # fill_value = used to fill missing values default = -1
    # usemask    = False --> returns a regular structured array
    # asrecarray = True --> output as np.recarray, False --> np.array
    wel_conc_dtype = list(np.full(len(wel_conc), "<f8"))
    record_addconc = rfn.append_fields(
        record_trimmed,
        component_name_l,
        wel_conc,
        dtypes=wel_conc_dtype,
        usemask=False,
        asrecarray=True,
    )

    # create new recarray and add boundname at the end
    new_dtype = record_addconc.dtype.descr + [("boundname", "O")]
    record_final = np.empty(record_addconc.shape, dtype=new_dtype)
    for name in record_addconc.dtype.names:
        record_final[name] = record_addconc[name]
    record_final["boundname"] = boundname

    return record_final
