"""ASR Reactive Transport Utilities
To complement functions in `flopy` and `mf6rtm`
"""

from pathlib import Path
import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
import flopy
import matplotlib.pyplot as plt


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

    sourcerecarray = [(gwf.wel.package_name, "AUX", component_name),
     (gwf.chd.package_name, "AUX", component_name),
     (gwf.ghb.package_name, "AUX", component_name)]
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
    """loads concentration for all solutes for full simulation length"""
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

def modify_chd_spd(
    record: np.recarray,
    component_name_l: list,
    chd_conc: list,
):
    # get boundname and save for later
    boundname = list(record["boundname"])
    # remove tds, temp and boundname using rfn.drop_fields
    record_trimmed = rfn.drop_fields(record, ["boundname"])  #'tds','temp'

    # add chd conc data
    # numpy.lib.recfunctions.append_fields(base, names, data, dtypes=None, fill_value=-1, usemask=True, asrecarray=False)
    # base       = original recarray you want to add fields to
    # names      = string for one filed or list of strings for new field name(s)
    # data       = 1D list/array of values for the new field(s). For multiple
    #              fields use a list of arrays, where len(array) = len(base)
    # dtypes     = dtype (string or dtype obj) or list of dtypes for names
    # fill_value = used to fill missing values default = -1
    # usemask    = False --> returns a regular structured array
    # asrecarray = True --> output as np.recarray, False --> np.array
    chd_conc_dtype = list(np.full(len(chd_conc), "<f8"))
    n = len(record_trimmed)
    chd_conc_arrays = [np.full(n, val, dtype=np.float64) for val in chd_conc] 
    record_addconc = rfn.append_fields(
        record_trimmed,
        component_name_l,
        chd_conc_arrays, # NOTE: must be array here with one entry for each records
        dtypes=chd_conc_dtype,
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

def modify_ghb_spd(
    record: np.recarray,
    component_name_l: list,
    ghb_conc: list,
):
    # get boundname and save for later
    boundname = list(record["boundname"])
    # remove tds, temp and boundname using rfn.drop_fields
    record_trimmed = rfn.drop_fields(record, ["boundname"])  #'tds','temp'

    # add chd conc data
    # numpy.lib.recfunctions.append_fields(base, names, data, dtypes=None, fill_value=-1, usemask=True, asrecarray=False)
    # base       = original recarray you want to add fields to
    # names      = string for one filed or list of strings for new field name(s)
    # data       = 1D list/array of values for the new field(s). For multiple
    #              fields use a list of arrays, where len(array) = len(base)
    # dtypes     = dtype (string or dtype obj) or list of dtypes for names
    # fill_value = used to fill missing values default = -1
    # usemask    = False --> returns a regular structured array
    # asrecarray = True --> output as np.recarray, False --> np.array
    ghb_conc_dtype = list(np.full(len(ghb_conc), "<f8"))
    n = len(record_trimmed)
    ghb_conc_arrays = [np.full(n, val, dtype=np.float64) for val in ghb_conc] 
    record_addconc = rfn.append_fields(
        record_trimmed,
        component_name_l,
        ghb_conc_arrays, # NOTE: must be array here with one entry for each records
        dtypes=ghb_conc_dtype,
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

def calc_Cr(sim, gwf, gwt, layer):
    ''' Calculates the Courant number in the x, y and z directions at each
        timestep and stores the max and sum of Cr_x, Cr_y, and Cr_z '''
    
    # sim   = MF6 simulation object
    # gwf   = MF6 groundwater flow model object

    ### get grid cell dimensions
    grid_type = gwf.get_grid_type()
    grid_package = gwf.get_package(grid_type.name)
    cell2D = grid_package.cell2d.get_data()
    cell2D_df = pd.DataFrame.from_records(cell2D, index='icell2d')
    vertices = grid_package.vertices.get_data()
    vertices_df = pd.DataFrame.from_records(vertices, index='iv')
    top = grid_package.top.array
    botm = grid_package.botm.array

    ### calculate surface area of each grid cell within a single layer
    for cell in range(len(cell2D_df)):
        ### get vertice coordinates to calculate dx and dy
        temp_cell_info = cell2D_df.iloc[[cell]]
        # read each vert_1 - 5
        icverts = temp_cell_info.filter(like="icvert_").iloc[0].to_list()
        # remove Nones
        icverts_clean = [int(v) for v in icverts if v is not None]
        # look up (x,y) and create x and y arrays
        x_l = vertices_df.loc[icverts_clean,"xv"].to_numpy()
        y_l = vertices_df.loc[icverts_clean,"yv"].to_numpy()
        # calculate dx and dy 
        dx_temp = (np.max(x_l)-np.min(x_l))
        dy_temp = (np.max(y_l)-np.min(y_l))
        #surface_area = (np.max(x_l)-np.min(x_l)) * (np.max(y_l) - np.min(y_l))
        cell2D_df.loc[cell,'dx'] = dx_temp
        cell2D_df.loc[cell,'dy'] = dy_temp

    # convert dx and dy to numpy array
    dx = cell2D_df['dx'].to_numpy()
    dy = cell2D_df['dy'].to_numpy()

    # calc layer dz for each layer
    # initialize dz array
    dz = np.zeros_like(botm)
    # for layer 1:
    dz[0] = top - botm[0]
    # for layers 2:nlay
    dz[1:] = botm[:-1] - botm[1:]

    ### read in specific discharge (flow through a cross section)
    bud_gwf = gwf.output.budget()
    spdis = bud_gwf.get_data(text="DATA-SPDIS")
    head = gwf.output.head().get_alldata()

    qx_l = []
    qy_l = []
    qz_l = []

    for t in range(len(head)):
        qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(
            spdis[0], gwf, head=head[0]
        )
        qx_l.append(qx)
        qy_l.append(qy)
        qz_l.append(qz)

    """qx, qy, qz are ndarrays of size (nlay, nrow, ncol) for a structured grid or 
    size (nlay, ncpl) for an unstructured grid. The sign of qy is such that the y 
    axis is considered to increase in the north direction. The sign of qz is such 
    that the z axis is considered to increase in the upward direction. Note: if a 
    head array is provided, inactive and dry cells are set to NaN."""

    ### calculate pore water velocity (q/n) (flow through the pores)
    mst = gwt.get_package('mst')
    porosity = mst.porosity.get_data()

    v_pw_x = np.array(qx_l)/porosity[layer][0]
    v_pw_y = np.array(qy_l)/porosity[layer][0]
    v_pw_z = np.array(qz_l)/porosity[layer][0]

    ### calculate Cr number (v_pore_water * dt / cell_dimention)
    Cr_sum = np.full_like(v_pw_x, np.nan)
    Cr_max = np.full_like(v_pw_x, np.nan)
    # read in tdis for perlen and nstp
    tdis_spd = sim.get_package("tdis").perioddata.get_data(full_data=True)
    dt_all = tdis_spd['perlen']/tdis_spd['nstp']

    t = 0
    for step in tdis_spd['nstp']:
        dt = dt_all[step]
        for s in range(step):
            Cr_x_temp = abs(v_pw_x[t]) * dt / dx
            Cr_y_temp = abs(v_pw_y[t]) * dt / dy
            Cr_z_temp = abs(v_pw_z[t]) * dt / dz
            Cr_sum[t] = np.round((Cr_x_temp + Cr_y_temp + Cr_z_temp),decimals=6)  
            Cr_max[t] = np.round(np.maximum(np.maximum(Cr_x_temp,Cr_y_temp),Cr_z_temp),decimals=6)
            t = t + 1

    print('Cr_sum > 1:  ' + str(np.where(Cr_sum>1.)))
    print('Cr_sum > 0.5:  ' + str(np.where(Cr_sum>0.5)))

    return Cr_sum, Cr_max


def plot_Cr_map_view(gwf,t_l,layer,Cr_type,Cr_data,vmin,vmax,f=101,extent=None):
    # plot Courant number in map view using FloPy PlotMapView
    # gwf     = flopy gwf model object
    # t_l     = list of timestep index (NOT actual time/days)
    # layer   = model layer to plot map view
    # Cr_type = string of what Cr values used (Cr_sum, Cr_max...)
    # Cr_data = np.array of calculated Cr number in shape [np.sum(tdis_spd['nstp']), nlay, ncpl]
    # f       = starting figure number
    # extent  = PlotMapView extent option: (xmin, xmax, ymin, ymax) 

    for t in t_l:
        fig = plt.figure(num=f,figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1, aspect="auto")
        ax.set_title(f"{Cr_type} at timestep index t=" + str(t))
        mapview = flopy.plot.PlotMapView(gwf, layer=layer, extent=extent)
        patch_collection = mapview.plot_array(Cr_data[t,:,:],vmin=vmin, vmax=vmax)
        linecollection = mapview.plot_grid()
        cb = plt.colorbar(patch_collection, shrink=0.75)
        plt.show()
        f = f + 1
    return 

def plot_cross_section(gwf,conc,times_c,layer,f,plot_bcs,line,c,vmin_l,vmax_l,t_l,component_name_l,extent=None,gwt_ic_conc_l=None):
    # plot a cross section along the line specified for each timestep in t_l
    # gwf      = flopy gwf model object
    # conc     = list of np.arrays of conc for each component with arrray shape [ntimes, nlay, 1, ncpl]
    # layer    = model layer to plot cross section for (0-based index) 
    # f        = starting figure number
    # plot_bcs = True/False for whether to plot BCs on map view showing
    # line     = array of x and y coordinates for start and end of line to plot cross section along
    # c        = index of component to plot xsection from component_name_l
    # vmin_l   = list of vmin for each component to use for plotting cross section
    # vmax_l   = list of vmax for each component to use for plotting cross section
    # t_l      = list of timestep index (actual time/days) to plot cross section for
    #          = NOTE: if t_l = "ic" then it will plot the initial conditions for the component instead of timesteps
    # extent   = PlotMapView extent option: (xmin, xmax, ymin, ymax)

    # create cross section line to plot cross section along
    # to plot a cross section with disv, you have to make a line to plot along
    # creates a plot showing where the line is on the grid to make the cross section plot
# create cross section line to plot cross section along
# to plot a cross section with disv, you have to make a line to plot along
# creates a plot showing where the line is on the grid to make the cross section plot
    fig = plt.figure(num=f,figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1, aspect="auto")
    ax.set_title("Vertex Model Grid (DISV) with cross sectional line")
    # use PlotMapView to plot a DISV (vertex) model
    mapview = flopy.plot.PlotMapView(gwf, layer=layer,extent=extent)
    if plot_bcs == True:
        mapview.plot_bc("WEL-1")
        mapview.plot_bc("CHD-1")
    linecollection = mapview.plot_grid()
    # plot the line over the model grid
    lc = plt.plot(line.T[0], line.T[1], "r--", lw=0.8)
    plt.show()
    f = f + 1

    # creates a cross section along the line specified above for each timestep in t_l
    # list of timestep index (NOT actual time/days)
    if t_l == "ic":
        fig = plt.figure(num=f,figsize=(6, 4))
        xsect = flopy.plot.PlotCrossSection(model=gwf, line={"line": line})
        patch_collection = xsect.plot_array(gwt_ic_conc_l[c], vmin=vmin_l[c], vmax=vmax_l[c])
        line_collection = xsect.plot_grid()
        cb = plt.colorbar(patch_collection, shrink=0.75)
        title = "Initial Conditions: " + component_name_l[c]
        plt.title(title)
        plt.show()
        f = f + 1
    else:
        for t in t_l:
            fig = plt.figure(num=f,figsize=(6, 4))
            xsect = flopy.plot.PlotCrossSection(model=gwf, line={"line": line})
            patch_collection = xsect.plot_array(conc[c][t, :, :, :], vmin=vmin_l[c], vmax=vmax_l[c])
            line_collection = xsect.plot_grid()
            cb = plt.colorbar(patch_collection, shrink=0.75)
            title = "Timestep: t=" + str(t) + ', '+ str(times_c[c][t]) + " days for " + component_name_l[c]
            plt.title(title)    
            plt.show()
            f = f + 1

def plot_conc_map_view(gwf,component_name,t_l,layer,conc,vmin,vmax,f,extent=None):
    # plot Courant number in map view using FloPy PlotMapView
    # gwf     = flopy gwf model object
    # t_l     = list of timestep index (NOT actual time/days)
    # layer   = model layer to plot map view
    # f       = starting figure number
    # extent  = PlotMapView extent option: (xmin, xmax, ymin, ymax)
    # conc    = array of concentration with shape [ntimes, nlay, 1, ncpl]

    for t in t_l:
        fig = plt.figure(num=f,figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1, aspect="auto")
        ax.set_title(f"{component_name} at timestep index t=" + str(t))
        mapview = flopy.plot.PlotMapView(gwf, layer=layer, extent=extent)
        patch_collection = mapview.plot_array(conc[t,:,:,:],vmin=vmin, vmax=vmax)
        linecollection = mapview.plot_grid()
        cb = plt.colorbar(patch_collection, shrink=0.75)
        plt.show()
        f = f + 1
    return 