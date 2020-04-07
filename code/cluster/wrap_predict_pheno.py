#!/usr/bin/env python
# coding: utf-8

# In[34]:


import os
import numpy as np
import subprocess

py_exec = '/cbica/home/parkesl/miniconda3/envs/neurodev_cs_predictive/bin/python'
py_script = '/cbica/home/parkesl/ResProjects/neurodev_cs_predictive/code/cluster/predict_pheno.py'
indir = '/cbica/home/parkesl/ResProjects/neurodev_cs_predictive/analysis_cubic/normative/t1Exclude/squeakycleanExclude/schaefer_200_streamlineCount_consist/ageAtScan1_Years+sex_adj/predict_pheno'

metrics = ['ac', 'mc']
phenos = ['Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg','AnxiousMisery','Externalizing','Fear']
algs = ['rr', 'krr_lin', 'krr_rbf']
seeds = np.arange(0,200)

print(len(seeds) * len(algs) * len(metrics) * len(phenos))

for seed in seeds:
    for alg in algs:
        for metric in metrics:
            for pheno in phenos:
                subprocess_str = '{0} {1} -x {2}/X.csv -y {2}/y.csv -seed {3} -alg {4} -metric {5} -pheno {6} -o {2}'.format(py_exec, py_script, indir, seed, alg, metric, pheno)
                # subprocess_str

                name = 's' + str(seed) + '_' + alg + '_' + metric + '_' + pheno
                qsub_call = 'qsub -N {0} -l h_vmem=1G,s_vmem=1G -pe threaded 1 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ '.format(name)

                os.system(qsub_call + subprocess_str)

