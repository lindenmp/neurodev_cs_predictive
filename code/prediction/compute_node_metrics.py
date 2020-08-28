#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Essentials
import os, sys, glob
import pandas as pd
import numpy as np
import nibabel as nib
import scipy.io as sio

# Stats
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import pingouin as pg

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


# In[2]:


from sklearn.linear_model import LinearRegression


# In[3]:


sys.path.append('/Users/lindenmp/Google-Drive-Penn/work/research_projects/neurodev_cs_predictive/code/func/')
from proj_environment import set_proj_env
sys.path.append('/Users/lindenmp/Google-Drive-Penn/work/misc_projects/pyfunc/')
from func import node_strength, ave_control, modal_control, consistency_thresh, rank_int, ave_control_alt, modal_control_alt


# In[4]:


exclude_str = 't1Exclude' # 't1Exclude' 'fsFinalExclude'
extra_str = '' # '_consist'
edge_weight = 'streamlineCount' # 'streamlineCount' 'fa' 'mean_streamlineLength' 'adc'
parc_str = 'schaefer' # 'schaefer' 'lausanne'
parc_scale = 200
parc_variant = 'orig' # 'orig' 'cortex_only'
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(exclude_str = exclude_str,
                                                                                        parc_str = parc_str, parc_scale = parc_scale,
                                                                                       extra_str = extra_str, edge_weight = edge_weight,
                                                                                       parc_variant = parc_variant)


# ### Setup output directory

# In[5]:


print(os.environ['MODELDIR'])
if not os.path.exists(os.environ['MODELDIR']): os.makedirs(os.environ['MODELDIR'])


# ### Data processing options

# In[6]:


if extra_str == '_consist':
    threshold = True
else:
    threshold = False
print(threshold)
    
vol_norm = False
if 'altcontrol' in os.environ['MODELDIR']:
    print('Using alt control code')
else:
    print('Using regular control code')


# ## Load train/test .csv and setup node .csv

# In[7]:


os.path.join(os.environ['TRTEDIR'])


# In[8]:


# Load data
df = pd.read_csv(os.path.join(os.environ['TRTEDIR'], 'df_pheno.csv'))
df.set_index(['bblid', 'scanid'], inplace = True)
print(df.shape)


# In[9]:


# Missing data file for this subject only for schaefer 200
if parc_scale == 200:
    df.drop(labels = (112598, 5161), inplace=True)


# In[10]:


# output dataframe
str_labels = ['str_' + str(i) for i in range(num_parcels)]
ac_labels = ['ac_' + str(i) for i in range(num_parcels)]
mc_labels = ['mc_' + str(i) for i in range(num_parcels)]

df_node = pd.DataFrame(index = df.index, columns = str_labels + ac_labels + mc_labels)
# df_node.insert(0, 'nuisance_sample', df['nuisance_sample'])

print(df_node.shape)


# ## Load in structural connectivity matrices

# In[11]:


# subject filter
subj_filt = np.zeros((df.shape[0],)).astype(bool)


# In[12]:


A = np.zeros((num_parcels, num_parcels, df.shape[0]))
for (i, (index, row)) in enumerate(df.iterrows()):
    file_name = os.environ['SC_NAME_TMP'].replace("scanid", str(index[1]))
    file_name = file_name.replace("bblid", str(index[0]))
    full_path = glob.glob(os.path.join(os.environ['SCDIR'], file_name))
    if i == 0: print(full_path)
    if len(full_path) > 0:
        mat_contents = sio.loadmat(full_path[0])
        a = mat_contents[os.environ['CONN_STR']]
        if parc_str == 'lausanne' and parc_variant == 'cortex_only':
            a = a[parcel_loc == 1,:]
            a = a[:,parcel_loc == 1]
        A[:,:,i] = a
    elif len(full_path) == 0:
        print(file_name + ': NOT FOUND')
        subj_filt[i] = True
        A[:,:,i] = np.full((num_parcels, num_parcels), np.nan)


# In[13]:


np.sum(subj_filt)


# In[14]:


if any(subj_filt):
    A = A[:,:,~subj_filt]
    df = df.loc[~subj_filt]
    df_node = df_node.loc[~subj_filt]
print(df_node.shape)


# ### Consistency thresholding

# In[15]:


if threshold:
    A_out, A_mask = consistency_thresh(A, thresh = 0.6)
    sns.heatmap(A_mask)
else:
    print('skipping..')
    A_out = A.copy()


# In[16]:


if threshold:
    np.all(np.sum(A_mask, axis = 1) > 0)


# ### Check if any subjects have disconnected nodes in A matrix

# In[17]:


# subject filter
subj_filt = np.zeros((df.shape[0],)).astype(bool)


# In[18]:


for i in range(A_out.shape[2]):
    if np.any(np.sum(A_out[:,:,i], axis = 1) == 0):
        subj_filt[i] = True


# In[19]:


np.sum(subj_filt)


# In[20]:


if any(subj_filt):
    A_out = A_out[:,:,~subj_filt]
    df = df.loc[~subj_filt]
    df_node = df_node.loc[~subj_filt]
print(df_node.shape)


# In[21]:


np.sum(df['averageManualRating'] == 2)


# In[22]:


np.sum(df['dti64QAManualScore'] == 2)


# ### Get streamline count and network density

# In[21]:


A_c = np.zeros((A_out.shape[2],))
A_d = np.zeros((A_out.shape[2],))
for i in range(A_out.shape[2]):
    A_c[i] = np.sum(np.triu(A_out[:,:,i]))
    A_d[i] = np.count_nonzero(np.triu(A_out[:,:,i]))/((A_out[:,:,i].shape[0]**2-A_out[:,:,i].shape[0])/2)
df.loc[:,'streamline_count'] = A_c
df.loc[:,'network_density'] = A_d


# ### Normalize A by regional volume

# In[22]:


if vol_norm:
    vol_ref = os.path.join(os.environ['DERIVSDIR'], 'Schaefer2018_'+str(parc_scale)+'_17Networks_PNC_2mm.nii.gz')
    print(vol_ref)
    img = nib.load(vol_ref)
    v = np.array(img.dataobj)
    v = v[v != 0]
    unique_elements, counts_elements = np.unique(v, return_counts=True)

    for i in range(num_parcels):
        for j in range(num_parcels):
            denom = counts_elements[i] + counts_elements[j]
            A_out[i,j,:] = A_out[i,j,:]/denom
else:
    print('skipping..')


# ### Compute node metrics

# In[23]:


# fc stored as 3d matrix, subjects of 3rd dim
S = np.zeros((df.shape[0], num_parcels))
AC = np.zeros((df.shape[0], num_parcels))
MC = np.zeros((df.shape[0], num_parcels))

for (i, (index, row)) in enumerate(df.iterrows()):
    S[i,:] = node_strength(A_out[:,:,i])
    if 'altcontrol' in os.environ['MODELDIR']:
        AC[i,:] = ave_control_alt(A_out[:,:,i])
        MC[i,:] = modal_control_alt(A_out[:,:,i])
    else:
        AC[i,:] = ave_control(A_out[:,:,i])
        MC[i,:] = modal_control(A_out[:,:,i])


df_node.loc[:,str_labels] = S
df_node.loc[:,ac_labels] = AC
df_node.loc[:,mc_labels] = MC


# ## Recalculate average control at different C params

# In[24]:


if 'altcontrol' in os.environ['MODELDIR']:
    c_params = np.linspace(0.75, 0.25, 3)
else:
    c_params = np.array([10, 100, 1000, 10000])

c_params


# In[25]:


# output dataframe
df_node_ac_overc = pd.DataFrame(index = df.index)

for c in c_params:
    print(c)
    ac_labels_new = ['ac_c' + str(c) + '_' + str(i) for i in range(num_parcels)]
    df_node_ac_temp = pd.DataFrame(index = df.index, columns = ac_labels_new)
    
    # fc stored as 3d matrix, subjects of 3rd dim
    AC = np.zeros((df.shape[0], num_parcels))
    for (i, (index, row)) in enumerate(df.iterrows()):
        if 'altcontrol' in os.environ['MODELDIR']:
            AC[i,:] = ave_control_alt(A_out[:,:,i], c = c)
        else:
            AC[i,:] = ave_control(A_out[:,:,i], c = c)

    df_node_ac_temp.loc[:,ac_labels_new] = AC
    df_node_ac_overc = pd.concat((df_node_ac_overc, df_node_ac_temp), axis = 1)


# # Save out raw data

# In[26]:


print(df_node.isna().any().any())
print(df_node_ac_overc.isna().any().any())


# In[27]:


os.environ['MODELDIR']


# In[28]:


# Save out
np.save(os.path.join(os.environ['MODELDIR'], 'A'), A)
np.save(os.path.join(os.environ['MODELDIR'], 'A_out'), A_out)

df_node.to_csv(os.path.join(os.environ['MODELDIR'], 'df_node_base.csv'))
df_node_ac_overc.to_csv(os.path.join(os.environ['MODELDIR'], 'df_node_ac_overc_base.csv'))

df.to_csv(os.path.join(os.environ['MODELDIR'], 'df_pheno.csv'))


# # Export for prediction

# ## Normalize

# ### Covariates

# In[29]:


covs = ['ageAtScan1', 'mprage_antsCT_vol_TBV', 'dti64MeanRelRMS', 'network_density', 'streamline_count']


# In[30]:


rank_r = np.zeros(len(covs),)

for i, cov in enumerate(covs):
    x = rank_int(df.loc[:,cov])
    rank_r[i] = sp.stats.spearmanr(df.loc[:,cov],x)[0]
    df.loc[:,cov] = x

print(np.sum(rank_r < 0.99))


# ### Node features

# In[31]:


rank_r = np.zeros(df_node.shape[1],)

for i, col in enumerate(df_node.columns):
    x = rank_int(df_node.loc[:,col])
    rank_r[i] = sp.stats.spearmanr(df_node.loc[:,col],x)[0]
    df_node.loc[:,col] = x

print(np.sum(rank_r < .99))


# In[32]:


rank_r = np.zeros(df_node_ac_overc.shape[1],)

for i, col in enumerate(df_node_ac_overc.columns):
    x = rank_int(df_node_ac_overc.loc[:,col])
    rank_r[i] = sp.stats.spearmanr(df_node_ac_overc.loc[:,col],x)[0]
    df_node_ac_overc.loc[:,col] = x

print(np.sum(rank_r < .99))


# ### Psychosis

# In[33]:


covs = ['ageAtScan1', 'sex', 'mprage_antsCT_vol_TBV', 'dti64MeanRelRMS']
phenos = ['Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg']
print(phenos)

# Create subdirectory for specific normative model -- labeled according to parcellation/resolution choices and covariates
outdir = os.path.join(os.environ['MODELDIR'], 'predict_psy')
print(outdir)
if not os.path.exists(outdir): os.mkdir(outdir);

df_node.to_csv(os.path.join(outdir, 'X.csv'))
df_node_ac_overc.to_csv(os.path.join(outdir, 'X_ac_c.csv'))
df.loc[:,phenos].to_csv(os.path.join(outdir, 'y.csv'))
df.loc[:,covs].to_csv(os.path.join(outdir, 'c.csv'))

covs = ['ageAtScan1', 'sex', 'mprage_antsCT_vol_TBV', 'dti64MeanRelRMS', 'streamline_count']
df.loc[:,covs].to_csv(os.path.join(outdir, 'c_sc.csv'))

