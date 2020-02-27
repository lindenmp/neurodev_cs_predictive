#!/usr/bin/env python
# coding: utf-8

# # Results, section 2:

# In[1]:


import os, sys
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
from sklearn.impute import SimpleImputer


# In[2]:


sys.path.append('/Users/lindenmp/Dropbox/Work/ResProjects/NormativeNeuroDev_CrossSec_DWI/code/func/')
from proj_environment import set_proj_env
from func import mark_outliers, get_cmap, run_corr, get_fdr_p, perc_dev, evd, summarise_network


# In[3]:


exclude_str = 't1Exclude'
parc_str = 'schaefer'
parc_scale = 200
primary_covariate = 'ageAtScan1'
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(exclude_str = exclude_str,
                                                                            parc_str = parc_str, parc_scale = parc_scale,
                                                                            primary_covariate = primary_covariate)


# In[4]:


os.environ['NORMATIVEDIR']


# In[5]:


metrics = ('ct', 'vol', 'str', 'ac', 'mc')
phenos = ('Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg','AnxiousMisery','Externalizing','Fear',
         'F1_Exec_Comp_Res_Accuracy', 'F2_Social_Cog_Accuracy', 'F3_Memory_Accuracy', 'F1_Complex_Reasoning_Efficiency',
          'F2_Memory.Efficiency', 'F3_Executive_Efficiency', 'F4_Social_Cognition_Efficiency',)


# ## Setup plots

# In[6]:


if not os.path.exists(os.environ['FIGDIR']): os.makedirs(os.environ['FIGDIR'])
os.chdir(os.environ['FIGDIR'])
sns.set(style='white', context = 'talk', font_scale = 1)
cmap = sns.color_palette("pastel", 3)


# ## Load data

# In[7]:


# Train
df_train = pd.read_csv(os.path.join(os.environ['NORMATIVEDIR'], 'train.csv'))
df_train.set_index(['bblid', 'scanid'], inplace = True); print(df_train.shape)
df_node_train = pd.read_csv(os.path.join(os.environ['NORMATIVEDIR'], 'resp_train.csv'))
df_node_train.set_index(['bblid', 'scanid'], inplace = True); print(df_node_train.shape)
z_cv = np.loadtxt(os.path.join(os.environ['NORMATIVEDIR'], 'cv/Z.txt'), delimiter = ' ').transpose()
df_z_cv = pd.DataFrame(data = z_cv, index = df_node_train.index, columns = df_node_train.columns); print(df_z_cv.shape)

# Test
df_test = pd.read_csv(os.path.join(os.environ['NORMATIVEDIR'], 'test.csv'))
df_test.set_index(['bblid', 'scanid'], inplace = True); print(df_test.shape)
df_node_test = pd.read_csv(os.path.join(os.environ['NORMATIVEDIR'], 'resp_test.csv'))
df_node_test.set_index(['bblid', 'scanid'], inplace = True); print(df_node_test.shape)
z = np.loadtxt(os.path.join(os.environ['NORMATIVEDIR'], 'Z.txt'), delimiter = ' ').transpose()
df_z_test = pd.DataFrame(data = z, index = df_node_test.index, columns = df_node_test.columns); print(df_z_test.shape)

# SMSE
smse = np.loadtxt(os.path.join(os.environ['NORMATIVEDIR'], 'smse.txt'), delimiter = ' ').transpose()
df_smse = pd.DataFrame(data = smse, index = df_node_test.columns)


# In[8]:


df = pd.concat((df_train,df_test), axis = 0)
df_node = pd.concat((df_node_train,df_node_test), axis = 0)
df_z = pd.concat((df_z_cv,df_z_test), axis = 0)


# # Characterizing the psychopathology phenotype data

# Let's have a look at our psychopathology phenotype data, which are the continous DVs for our predictive model

# In[9]:


print('N:', df.shape[0])


# In[10]:


# How much missing data have I got in the phenotypes?
for pheno in phenos:
    print('No. of NaNs for ' + pheno + ':', df.loc[:,pheno].isna().sum())


# In[11]:


imputer = SimpleImputer(missing_values=np.nan, strategy='median')


# In[12]:


imputer.fit(df.loc[:,phenos])
df.loc[:,phenos] = imputer.transform(df.loc[:,phenos])


# In[13]:


# How much missing data have I got in the phenotypes?
for pheno in phenos:
    print('No. of NaNs for ' + pheno + ':', df.loc[:,pheno].isna().sum())


# In[14]:


# my_bool = df.loc[:,phenos].isna().any(axis = 1)

# df = df.loc[~my_bool,:]
# df_node = df_node.loc[~my_bool,:]
# df_z = df_z.loc[~my_bool,:]

# print('N:', df.shape[0])


# In[15]:


df.columns


# In[16]:


f, ax = plt.subplots(1,len(phenos))
f.set_figwidth(len(phenos)*5)
f.set_figheight(5)

if len(phenos) == 1:
    sns.distplot(df.loc[:,phenos[0]], ax = ax, color = cmap[0])
    ax.set_xlabel(phenos[0])
else:
    for i, pheno in enumerate(phenos):
        sns.distplot(df.loc[:,pheno], ax = ax[i], color = cmap[0])
        ax[i].set_xlabel(pheno)     


# In[17]:


for pheno in phenos:
    df.loc[:,pheno] = sp.stats.yeojohnson(df.loc[:,pheno])[0]
#     df.loc[:,pheno] = np.log(df.loc[:,pheno] + (df.loc[:,pheno].abs().max()+1))
#     df.loc[:,pheno] = (df.loc[:,pheno] - df.loc[:,pheno].mean())/df.loc[:,pheno].std()  


# In[18]:


f, ax = plt.subplots(1,len(phenos))
f.set_figwidth(len(phenos)*5)
f.set_figheight(5)

if len(phenos) == 1:
    sns.distplot(df.loc[:,phenos[0]], ax = ax, color = cmap[0])
    ax.set_xlabel(phenos[0])
else:
    for i, pheno in enumerate(phenos):
        sns.distplot(df.loc[:,pheno], ax = ax[i], color = cmap[0])
        ax[i].set_xlabel(pheno)     


# ### Regress age/sex out of DVs

# In[19]:


df_nuis = df.loc[:,[primary_covariate,'sex_adj']]
df_nuis = sm.add_constant(df_nuis)

# phenos
mdl = sm.OLS(df.loc[:,phenos], df_nuis).fit()
y_pred = mdl.predict(df_nuis)
y_pred.columns = phenos
df.loc[:,phenos] = df.loc[:,phenos] - y_pred

# # df_node
# cols = df_node.columns
# mdl = sm.OLS(df_node.loc[:,cols], df_nuis).fit()
# y_pred = mdl.predict(df_nuis)
# y_pred.columns = cols
# df_node.loc[:,cols] = df_node.loc[:,cols] - y_pred


# In[20]:


f, ax = plt.subplots(1,len(phenos))
f.set_figwidth(len(phenos)*5)
f.set_figheight(5)

if len(phenos) == 1:
    sns.distplot(df.loc[:,phenos[0]], ax = ax, color = cmap[0])
    ax.set_xlabel(phenos[0])
else:
    for i, pheno in enumerate(phenos):
        sns.distplot(df.loc[:,pheno], ax = ax[i], color = cmap[0])
        ax[i].set_xlabel(pheno)     


# In[21]:


df.loc[:,phenos].head()


# In[22]:


for i, pheno in enumerate(phenos):
    x = df.loc[:,pheno]
    my_med = np.median(x)
    mad = np.median(abs(x - my_med))/1.4826
    print(mad)


# # Feature selection

# ### Normative model performance

# In[23]:


smse_thresh = 1
smse_filter = df_smse.values < smse_thresh
smse_filter = smse_filter.reshape(-1)
smse_filter.sum()


# Combine all the filters into one

# In[24]:


# region_filter = np.logical_and(age_filter,smse_filter)
region_filter = smse_filter
region_filter.sum()


# In[25]:


print('regions per metric')
for metric in metrics:
    print(metric + ': ' + str(df_z.loc[:,region_filter].filter(regex = metric).shape[1]))


# In[26]:


df.loc[:,phenos].to_csv(os.path.join(os.environ['NORMATIVEDIR'], 'df.csv'))
df_node.loc[:,region_filter].to_csv(os.path.join(os.environ['NORMATIVEDIR'], 'df_node.csv'))
df_z.loc[:,region_filter].to_csv(os.path.join(os.environ['NORMATIVEDIR'], 'df_z.csv'))


# # Feature summaries

# Alternatively could be to collapse over brain regions into single summary measures. There are a few obvious ways to do this: mean, extreme value stats. Let's look at a few!

# In[27]:


df_node_summary = pd.DataFrame(index = df_node.index)
for metric in metrics:
    df_node_summary[metric+'_node_mean'] = df_node.loc[:,region_filter].filter(regex = metric, axis = 1).mean(axis = 1)
    df_node_summary[metric+'_z_mean'] = df_z.loc[:,region_filter].filter(regex = metric, axis = 1).mean(axis = 1)
    df_node_summary[metric+'_z_evd_pos'] = evd(df_z.loc[:,region_filter].filter(regex = metric, axis = 1), sign = 'pos')
    df_node_summary[metric+'_z_evd_neg'] = evd(df_z.loc[:,region_filter].filter(regex = metric, axis = 1), sign = 'neg')


# In[28]:


my_bool = df_node_summary.isna().any()
my_bool


# In[29]:


df_node_summary = df_node_summary.loc[:,~my_bool]


# In[30]:


df_node_summary.head()


# In[31]:


df_node_summary.to_csv(os.path.join(os.environ['NORMATIVEDIR'], 'df_node_summary.csv'))


# How do the summaries relate?

# In[32]:


R = pd.DataFrame(index = df_node_summary.columns, columns = df_node_summary.columns)

for i_col in df_node_summary.columns:
    for j_col in df_node_summary.columns:
        R.loc[i_col,j_col] = sp.stats.pearsonr(df_node_summary[i_col],df_node_summary[j_col])[0]


# In[33]:


f, ax = plt.subplots(1)
f.set_figwidth(10)
f.set_figheight(10)

sns.heatmap(R.astype(float), annot = False, center = 0, vmax = 1, vmin = -1, ax = ax, square = True)

