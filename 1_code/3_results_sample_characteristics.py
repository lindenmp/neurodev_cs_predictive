#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Essentials
import os, sys, glob
import pandas as pd
import numpy as np
import nibabel as nib
import scipy.io as sio
from tqdm import tqdm

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


sys.path.append('/Users/lindenmp/Google-Drive-Penn/work/research_projects/neurodev_cs_predictive/1_code/')
from func import set_proj_env, my_get_cmap, get_fdr_p


# In[3]:


parc_str = 'schaefer' # 'schaefer' 'lausanne' 'glasser'
parc_scale = 200 # 200/400 | 125 | 360
edge_weight = 'streamlineCount'
parcel_names, parcel_loc, drop_parcels, num_parcels = set_proj_env(parc_str = parc_str, parc_scale = parc_scale, edge_weight = edge_weight)


# In[4]:


# output file prefix
outfile_prefix = parc_str+'_'+str(parc_scale)+'_'+edge_weight+'_'
outfile_prefix


# ### Setup directory variables

# In[5]:


print(os.environ['PIPELINEDIR'])
if not os.path.exists(os.environ['PIPELINEDIR']): os.makedirs(os.environ['PIPELINEDIR'])


# In[6]:


figdir = os.path.join(os.environ['OUTPUTDIR'], 'figs')
print(figdir)
if not os.path.exists(figdir): os.makedirs(figdir)


# In[7]:


phenos = ['Psychosis_Positive','Psychosis_NegativeDisorg','Overall_Psychopathology']
phenos_label = ['Psychosis (positive)','Psychosis (negative)','Overall psychopathology']
phenos_short = ['Psy. (pos)','Psy. (neg)','Ov. psy.']

metrics = ['str', 'ac', 'bc', 'cc', 'sgc']
metrics_label = ['Strength', 'Average controllability', 'Betweenness centrality', 'Closeness centrality', 'Subgraph centrality']


# ## Setup plots

# In[8]:


if not os.path.exists(figdir): os.makedirs(figdir)
os.chdir(figdir)
sns.set(style='white', context = 'paper', font_scale = 1)
sns.set_style({'font.family':'sans-serif', 'font.sans-serif':['Public Sans']})
cmap = my_get_cmap('pair')


# ## Load data

# In[9]:


df = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '1_compute_node_features', 'store', outfile_prefix+'df.csv'))
df.set_index(['bblid', 'scanid'], inplace = True)

df_node = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '1_compute_node_features', 'store', outfile_prefix+'df_node.csv'))
df_node.set_index(['bblid', 'scanid'], inplace = True)

df_node_ac_i2 = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '1_compute_node_features', 'store', outfile_prefix+'df_node_ac_i2.csv'))
df_node_ac_i2.set_index(['bblid', 'scanid'], inplace = True)

df_node_ac_overc = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '1_compute_node_features', 'store', outfile_prefix+'df_node_ac_overc.csv'))
df_node_ac_overc.set_index(['bblid', 'scanid'], inplace = True)

df_node_ac_overc_i2 = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '1_compute_node_features', 'store', outfile_prefix+'df_node_ac_overc_i2.csv'))
df_node_ac_overc_i2.set_index(['bblid', 'scanid'], inplace = True)

c = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '1_compute_node_features', 'out', outfile_prefix+'c.csv'))
c.set_index(['bblid', 'scanid'], inplace = True); print(c.shape)


# In[10]:


print(np.sum(df_node.filter(regex = 'ac').corrwith(df_node_ac_i2, method='spearman') < 0.9999))
print(np.sum(df_node_ac_overc.corrwith(df_node_ac_overc_i2, method='spearman') < 0.9999))
print(np.sum(df_node.filter(regex = 'ac').corrwith(df_node_ac_i2, method='pearson') < 0.9999))
print(np.sum(df_node_ac_overc.corrwith(df_node_ac_overc_i2, method='pearson') < 0.9999))


# In[11]:


df['sex'].unique()


# In[12]:


print(np.sum(df.loc[:,'sex'] == 1))
print((np.sum(df.loc[:,'sex'] == 1)/df.shape[0]) * 100)


# In[13]:


print(np.sum(df.loc[:,'sex'] == 2))
print((np.sum(df.loc[:,'sex'] == 2)/df.shape[0]) * 100)


# In[14]:


print(df['ageAtScan1_Years'].mean())
print(c['ageAtScan1'].mean())
print(np.sum(c['ageAtScan1'] < c['ageAtScan1'].mean()))
print(c.shape[0]-np.sum(c['ageAtScan1'] <= c['ageAtScan1'].mean()))


# In[15]:


df['ageAtScan1_Years'].std()


# ### Sex

# In[16]:


stats = pd.DataFrame(index = phenos, columns = ['test_stat', 'pval'])

for i, pheno in enumerate(phenos):
    x = df.loc[df.loc[:,'sex'] == 1,pheno]
    y = df.loc[df.loc[:,'sex'] == 2,pheno]
    
    test_output = sp.stats.ttest_ind(x,y)
    stats.loc[pheno,'test_stat'] = test_output[0]
    stats.loc[pheno,'pval'] = test_output[1]
    
stats.loc[:,'pval_corr'] = get_fdr_p(stats.loc[:,'pval'])
stats.loc[:,'sig'] = stats.loc[:,'pval_corr'] < 0.05

stats


# In[17]:


f, ax = plt.subplots(1,len(phenos))
f.set_figwidth(len(phenos)*2.5)
f.set_figheight(2.5)

# sex: 1=male, 2=female
for i, pheno in enumerate(phenos):
    x = df.loc[df.loc[:,'sex'] == 1,pheno]
    sns.distplot(x, ax = ax[i], label = 'male')

    y = df.loc[df.loc[:,'sex'] == 2,pheno]
    sns.distplot(y, ax = ax[i], label = 'female')
    
    if i == 0:
        ax[i].legend()
    ax[i].set_xlabel(pheno)

    if stats.loc[pheno,'sig']:
        ax[i].set_title('t-stat:' + str(np.round(stats.loc[pheno,'test_stat'],2)) + ', p-value: ' + str(np.round(stats.loc[pheno,'pval_corr'],4)), fontweight="bold")
    else:
        ax[i].set_title('t-stat:' + str(np.round(stats.loc[pheno,'test_stat'],2)) + ', p-value: ' + str(np.round(stats.loc[pheno,'pval_corr'],4)))
        
f.savefig(outfile_prefix+'symptoms_distributions_sex.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# ### Age

# In[18]:


stats = pd.DataFrame(index = phenos, columns = ['r', 'pval'])

x = df['ageAtScan1_Years']
for i, pheno in enumerate(phenos):
    y = df[pheno]
    r,p = sp.stats.pearsonr(x,y)
    
    stats.loc[pheno,'r'] = r
    stats.loc[pheno,'pval'] = p
    
stats.loc[:,'pval_corr'] = get_fdr_p(stats.loc[:,'pval'])
stats.loc[:,'sig'] = stats.loc[:,'pval_corr'] < 0.05

stats


# In[19]:


f, ax = plt.subplots(1,len(phenos))
f.set_figwidth(len(phenos)*2.5)
f.set_figheight(2.5)

x = df['ageAtScan1_Years']
for i, pheno in enumerate(phenos):
    y = df[pheno]
    sns.regplot(x, y, ax=ax[i], scatter=False)
    ax[i].scatter(x, y, color='gray', s=5, alpha=0.5)
    
    if stats.loc[pheno,'sig']:
        ax[i].set_title('r:' + str(np.round(stats.loc[pheno,'r'],2)) + ', p-value: ' + str(np.round(stats.loc[pheno,'pval_corr'],4)), fontweight="bold")
    else:
        ax[i].set_title('r:' + str(np.round(stats.loc[pheno,'r'],2)) + ', p-value: ' + str(np.round(stats.loc[pheno,'pval_corr'],4)))
    
f.savefig(outfile_prefix+'symptoms_correlations_age.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# ### DWI data quality

# In[20]:


# 'dti64MeanRelRMS', 'dti64Tsnr', 'dti64Outmax', 'dti64Outmean',


# In[21]:


f, ax = plt.subplots()
f.set_figwidth(2)
f.set_figheight(2)
x = df['dti64MeanRelRMS']
sns.distplot(x, ax = ax)
ax.set_xlabel('In-scanner motion \n(mean relative framewise displacement)')
ax.set_ylabel('Counts')
ax.tick_params(pad = -2)

textstr = 'median = {:.2f}\nmean = {:.2f}\nstd = {:.2f}'.format(x.median(), x.mean(), x.std())
ax.text(0.975, 0.975, textstr, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right')

f.savefig(outfile_prefix+'inscanner_motion.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# In[22]:


f, ax = plt.subplots()
f.set_figwidth(2)
f.set_figheight(2)
x = df['dti64Tsnr']
sns.distplot(x, ax = ax)
ax.set_xlabel('Temporal signal to noise ratio')
ax.set_ylabel('Counts')
ax.tick_params(pad = -2)

textstr = 'median = {:.2f}\nmean = {:.2f}\nstd = {:.2f}'.format(x.median(), x.mean(), x.std())
ax.text(0.05, 0.975, textstr, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='left')

f.savefig(outfile_prefix+'dwi_tsnr.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# In[23]:


x = df['dti64MeanRelRMS']
# x = df['dti64Tsnr']

stats = pd.DataFrame(index = df_node.columns, columns = ['r','p'])

for col in df_node.columns:
    r = sp.stats.spearmanr(x, df_node.loc[:,col])
    stats.loc[col,'r'] = r[0]
    stats.loc[col,'p'] = r[1]

f, ax = plt.subplots(1,len(metrics))
f.set_figwidth(len(metrics)*1.5)
f.set_figheight(1.5)

for i, metric in enumerate(metrics):
    sns.distplot(stats.filter(regex = metric, axis = 0)['r'], ax = ax[i])
    
    ax[i].set_title(metrics_label[i])
    if i == 2: ax[i].set_xlabel('QC-SC (Spearman\'s rho)')
    else: ax[i].set_xlabel('')
    if i == 0: ax[i].set_ylabel('Counts')
    ax[i].tick_params(pad = -2)
    
    qc_sc = np.sum(stats.filter(regex = metric, axis = 0)['p']<.05)/num_parcels*100
    textstr = '{:.0f}%'.format(qc_sc)
    ax[i].text(0.975, 0.975, textstr, transform=ax[i].transAxes,
            verticalalignment='top', horizontalalignment='right')
    
f.savefig(outfile_prefix+'qc_sc.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)


# ### Diagnostic table

# In[24]:


# to_screen = ['goassessSmryPsy', 'goassessSmryMood', 'goassessSmryEat', 'goassessSmryAnx', 'goassessSmryBeh']
# counts = np.sum(df.loc[:,to_screen] == 4)
# print(counts)
# print(counts/df.shape[0]*100)


# In[25]:


df['goassessDxpmr4_bin'] = df.loc[:,'goassessDxpmr4'] == '4PS'
df['goassessDxpmr4_bin'] = df['goassessDxpmr4_bin'].astype(int)*4


# In[26]:


to_screen = ['goassessDxpmr4_bin','goassessSmryMan', 'goassessSmryDep', 'goassessSmryBul', 'goassessSmryAno', 'goassessSmrySoc',
             'goassessSmryPan', 'goassessSmryAgr', 'goassessSmryOcd', 'goassessSmryPtd', 'goassessSmryAdd',
            'goassessSmryOdd', 'goassessSmryCon']
counts = np.sum(df.loc[:,to_screen] == 4)
print(counts)
print(counts/df.shape[0]*100)


# In[27]:


to_keep = counts[counts >= 50].index
list(to_keep)


# In[28]:


counts[counts >= 50]


# In[29]:


my_xticklabels = ['Psychosis spectrum (n=303)',
                 'Depression (n=156)',
                 'Social anxiety disorder (n=261)',
                 'Agoraphobia (n=61)',
                 'PTSD (n=136)',
                 'ADHD (n=168)',
                 'ODD (n=353)',
                 'Conduct disorder (n=85)']


# In[30]:


f, ax = plt.subplots(1,len(phenos))
f.set_figwidth(len(phenos)*2.5)
f.set_figheight(2)

for i, pheno in enumerate(phenos):
    mean_scores = np.zeros(len(to_keep))
    for j, diagnostic_score in enumerate(to_keep):
        idx = df.loc[:,diagnostic_score] == 4
        mean_scores[j] = df.loc[idx,pheno].mean()
    
    ax[i].bar(x = np.arange(0,len(mean_scores)), height = mean_scores, color = 'w', edgecolor = 'k', linewidth = 1.5)
    ax[i].set_ylim([-.2,1.2])
    ax[i].set_xticks(np.arange(0,len(mean_scores)))
    ax[i].set_xticklabels(my_xticklabels, rotation = 90)
    ax[i].tick_params(pad = -2)
    ax[i].set_title(phenos_label[i])
    if i == 1:
        ax[i].set_xlabel('Psychopathology group')
    if i == 0:
        ax[i].set_ylabel('Factor score (z)')
    
f.savefig(outfile_prefix+'symptom_dimensions_groups.svg', dpi = 300, bbox_inches = 'tight', pad_inches = 0)

