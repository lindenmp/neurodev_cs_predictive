#!/usr/bin/env python
# coding: utf-8

# ## Submit jobs to cubic

# In[ ]:


import os
import numpy as np
import subprocess
import json

py_exec = '/cbica/home/parkesl/miniconda3/envs/neurodev_cs_predictive/bin/python'
py_script = '/cbica/home/parkesl/ResProjects/neurodev_cs_predictive/code/cluster/predict_pheno.py'
indir = '/cbica/home/parkesl/ResProjects/neurodev_cs_predictive/analysis_cubic/normative/t1Exclude/squeakycleanExclude/schaefer_200_streamlineCount_consist/ageAtScan1_Years+sex_adj/predict_pheno'

metrics = ['ac', 'mc']
phenos = ['Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg','AnxiousMisery','Externalizing','Fear']
algs = ['rr', 'krr_lin', 'krr_rbf']
seeds = np.arange(0,200)

num_seeds = len(seeds)
num_algs = len(algs)
num_metrics = len(metrics)
num_phenos = len(phenos)

print(num_seeds * num_algs * num_metrics * num_phenos)


# In[ ]:


for seed in seeds:
    for alg in algs:
        for metric in metrics:
            for pheno in phenos:
                subprocess_str = '{0} {1} -x {2}/X.csv -y {2}/y.csv -seed {3} -alg {4} -metric {5} -pheno {6} -o {2}'.format(py_exec, py_script, indir, seed, alg, metric, pheno)

                name = 's' + str(seed) + '_' + alg + '_' + metric + '_' + pheno
                qsub_call = 'qsub -N {0} -l h_vmem=1G,s_vmem=1G -pe threaded 1 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ '.format(name)

                os.system(qsub_call + subprocess_str)


# ## Assemble outputs

# In[ ]:


nested_score_mean = np.zeros((num_seeds, num_algs, num_metrics, num_phenos))
non_nested_r2 = np.zeros((num_seeds, num_algs, num_metrics, num_phenos))
non_nested_mse = np.zeros((num_seeds, num_algs, num_metrics, num_phenos))
non_nested_mae = np.zeros((num_seeds, num_algs, num_metrics, num_phenos))
non_nested_corr = np.zeros((num_seeds, num_algs, num_metrics, num_phenos))

for s, seed in enumerate(seeds):
    print(seed)
    for a, alg in enumerate(algs):
        for m, metric in enumerate(metrics):
            for p, pheno in enumerate(phenos):
                nested_score = np.loadtxt(os.path.join(indir, 'split_'+str(seed), alg + '_' + metric + '_' + pheno, 'nested_score.csv'))
                nested_score_mean[s,a,m,p] = nested_score.mean()
                
                best_scores = json.load(open(os.path.join(indir, 'split_'+str(seed), alg + '_' + metric + '_' + pheno, 'best_scores.json')))
                non_nested_r2[s,a,m,p] = best_scores['r2']
                non_nested_mse[s,a,m,p] = best_scores['mse']
                non_nested_mae[s,a,m,p] = best_scores['mae']
                non_nested_corr[s,a,m,p] = best_scores['corr']

np.save(os.path.join(indir, 'nested_score_mean'), nested_score_mean)
np.save(os.path.join(indir, 'non_nested_r2'), non_nested_r2)
np.save(os.path.join(indir, 'non_nested_mse'), non_nested_mse)
np.save(os.path.join(indir, 'non_nested_mae'), non_nested_mae)
np.save(os.path.join(indir, 'non_nested_corr'), non_nested_corr)

