# Linden Parkes, 2020
# lindenmp@seas.upenn.edu

# Essentials
import os, sys, glob
import pandas as pd
import numpy as np
import nibabel as nib

# Stats
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import pingouin as pg

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

def set_proj_env(parc_str = 'schaefer', parc_scale = 200, edge_weight = 'streamlineCount', extra_str = ''):

    # Project root directory
    projdir = '/Users/lindenmp/Google-Drive-Penn/work/research_projects/neurodev_cs_predictive'; os.environ['PROJDIR'] = projdir

    # Data directory
    datadir = os.path.join(projdir, '0_data'); os.environ['DATADIR'] = datadir

    # Imaging derivatives
    derivsdir = os.path.join('/Volumes/work_ssd/research_data/PNC/'); os.environ['DERIVSDIR'] = derivsdir

    # Pipeline directory
    pipelinedir = os.path.join(projdir, '2_pipeline'); os.environ['PIPELINEDIR'] = pipelinedir

    # Output directory
    outputdir = os.path.join(projdir, '3_output'); os.environ['OUTPUTDIR'] = outputdir

    # Parcellation specifications
    # Names of parcels
    if parc_str == 'schaefer': parcel_names = np.genfromtxt(os.path.join(projdir, 'figs_support/labels/schaefer' + str(parc_scale) + 'NodeNames.txt'), dtype='str')
    # vector describing whether rois belong to cortex (1) or subcortex (0)
    if parc_str == 'schaefer': parcel_loc = np.loadtxt(os.path.join(projdir, 'figs_support/labels/schaefer' + str(parc_scale) + 'NodeNames_loc.txt'), dtype='int')

    drop_parcels = []
    num_parcels = parcel_names.shape[0]

    if parc_str == 'schaefer':
        scdir = os.path.join(derivsdir, 'processedData/diffusion/deterministic_20171118'); os.environ['SCDIR'] = scdir
        sc_name_tmp = 'bblid/*xscanid/tractography/connectivity/bblid_*xscanid_SchaeferPNC_' + str(parc_scale) + '_dti_' + edge_weight + '_connectivity.mat'; os.environ['SC_NAME_TMP'] = sc_name_tmp

        os.environ['CONN_STR'] = 'connectivity'

        rstsdir = os.path.join(derivsdir, 'processedData/restbold/restbold_201607151621'); os.environ['RSTSDIR'] = rstsdir
        if parc_scale == 200:
            rsts_name_tmp = 'bblid/*xscanid/net/Schaefer' + str(parc_scale) + 'PNC/bblid_*xscanid_Schaefer' + str(parc_scale) + 'PNC_ts.1D'; os.environ['RSTS_NAME_TMP'] = rsts_name_tmp
        elif parc_scale == 400:
            rsts_name_tmp = 'bblid/*xscanid/net/SchaeferPNC/bblid_*xscanid_SchaeferPNC_ts.1D'; os.environ['RSTS_NAME_TMP'] = rsts_name_tmp

    return parcel_names, parcel_loc, drop_parcels, num_parcels


def my_get_cmap(which_type = 'qual1', num_classes = 8):
    # Returns a nice set of colors to make a nice colormap using the color schemes
    # from http://colorbrewer2.org/
    #
    # The online tool, colorbrewer2, is copyright Cynthia Brewer, Mark Harrower and
    # The Pennsylvania State University.

    if which_type == 'linden':
        cmap_base = np.array([[255,105,97],[97,168,255],[178,223,138],[117,112,179],[255,179,71]])
    elif which_type == 'pair':
        cmap_base = np.array([[124,230,199],[255,169,132]])
    elif which_type == 'qual1':
        cmap_base = np.array([[166,206,227],[31,120,180],[178,223,138],[51,160,44],[251,154,153],[227,26,28],
                            [253,191,111],[255,127,0],[202,178,214],[106,61,154],[255,255,153],[177,89,40]])
    elif which_type == 'qual2':
        cmap_base = np.array([[141,211,199],[255,255,179],[190,186,218],[251,128,114],[128,177,211],[253,180,98],
                            [179,222,105],[252,205,229],[217,217,217],[188,128,189],[204,235,197],[255,237,111]])
    elif which_type == 'seq_red':
        cmap_base = np.array([[255,245,240],[254,224,210],[252,187,161],[252,146,114],[251,106,74],
                            [239,59,44],[203,24,29],[165,15,21],[103,0,13]])
    elif which_type == 'seq_blu':
        cmap_base = np.array([[247,251,255],[222,235,247],[198,219,239],[158,202,225],[107,174,214],
                            [66,146,198],[33,113,181],[8,81,156],[8,48,107]])
    elif which_type == 'redblu_pair':
        cmap_base = np.array([[222,45,38],[49,130,189]])
    elif which_type == 'yeo17':
        cmap_base = np.array([[97,38,107], # VisCent
                            [194,33,39], # VisPeri
                            [79,130,165], # SomMotA
                            [44,181,140], # SomMotB
                            [75,148,72], # DorsAttnA
                            [23,116,62], # DorsAttnB
                            [149,77,158], # SalVentAttnA
                            [222,130,177], # SalVentAttnB
                            [75,87,61], # LimbicA
                            [149,166,110], # LimbicB
                            [210,135,47], # ContA
                            [132,48,73], # ContB
                            [92,107,131], # ContC
                            [218,221,50], # DefaultA
                            [175,49,69], # DefaultB
                            [41,38,99], # DefaultC
                            [53,75,158] # TempPar
                            ])
    elif which_type == 'yeo17_downsampled':
        cmap_base = np.array([[97,38,107], # VisCent
                            [79,130,165], # SomMotA
                            [75,148,72], # DorsAttnA
                            [149,77,158], # SalVentAttnA
                            [75,87,61], # LimbicA
                            [210,135,47], # ContA
                            [218,221,50], # DefaultA
                            [53,75,158] # TempPar
                            ])

    if cmap_base.shape[0] > num_classes: cmap = cmap_base[0:num_classes]
    else: cmap = cmap_base

    cmap = cmap / 255

    return cmap


def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2*c + 1)
    return sp.stats.norm.ppf(x)


def rank_int(series, c=3.0/8):
    # Check input
    assert(isinstance(series, pd.Series))
    assert(isinstance(c, float))

    # Set seed
    np.random.seed(123)

    # Drop NaNs
    series = series.loc[~pd.isnull(series)]

    # Get rank, ties are averaged
    rank = sp.stats.rankdata(series, method="average")

    # Convert numpy array back to series
    rank = pd.Series(rank, index=series.index)

    # Convert rank to normal distribution
    transformed = rank.apply(rank_to_normal, c=c, n=len(rank))
    
    return transformed