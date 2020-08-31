#!/usr/bin/env python
# coding: utf-8

# # Submit jobs to cubic

# In[ ]:


import os
import numpy as np
import subprocess
import json

py_exec = '/cbica/home/parkesl/miniconda3/envs/neurodev_cs_predictive/bin/python'

my_str = 'schaefer_200_streamlineCount'
# my_str = 'schaefer_400_streamlineCount'

indir = '/cbica/home/parkesl/research_projects/neurodev_cs_predictive/2_pipeline/1_compute_node_features/out/'+my_str+'_'
outdir = '/cbica/home/parkesl/research_projects/neurodev_cs_predictive/2_pipeline/3_prediction/out/'+my_str+'_'
phenos = ['Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg']

print(indir)
print(phenos)
    
metrics = ['str', 'ac']
algs = ['rr', 'krr_rbf']
scores = ['corr', 'rmse']

num_algs = len(algs)
num_metrics = len(metrics)
num_phenos = len(phenos)
num_scores = len(scores)

print(num_algs * num_metrics * num_phenos * num_scores)


# ## Random splits cross-val (nuis, no param optimization)

# In[ ]:


py_script = '/cbica/home/parkesl/research_projects/neurodev_cs_predictive/1_code/cluster/predict_symptoms_rcv_nuis.py'
modeldir = outdir+'predict_symptoms_rcv_nuis'

for alg in algs:
    for metric in metrics:
        for pheno in phenos:
            for score in scores:
                subprocess_str = '{0} {1} -x {2}X.csv -y {2}y.csv -c {2}c.csv -alg {3} -metric {4} -pheno {5} -score {6} -o {7}'.format(py_exec, py_script, indir, alg, metric, pheno, score, modeldir)

                name = 'prim' + '_' + alg + '_' + metric[0] + '_' + pheno[0] + '_' + score[0]
                qsub_call = 'qsub -N {0} -l h_vmem=1G,s_vmem=1G -pe threaded 4 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ '.format(name)

                os.system(qsub_call + subprocess_str)


# ### Over c

# In[ ]:


metrics = ['ac_c10', 'ac_c100', 'ac_c1000', 'ac_c10000']
py_script = '/cbica/home/parkesl/research_projects/neurodev_cs_predictive/1_code/cluster/predict_symptoms_rcv_nuis.py'
modeldir = outdir+'predict_symptoms_rcv_nuis'

for alg in algs:
    for metric in metrics:
        for pheno in phenos:
            for score in scores:
                subprocess_str = '{0} {1} -x {2}X_ac_c.csv -y {2}y.csv -c {2}c.csv -alg {3} -metric {4} -pheno {5} -score {6} -o {7}'.format(py_exec, py_script, indir, alg, metric, pheno, score, modeldir)

                name = 'prim' + '_' + alg + '_' + metric[0] + '_' + pheno[0] + '_' + score[0]
                qsub_call = 'qsub -N {0} -l h_vmem=1G,s_vmem=1G -pe threaded 2 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ '.format(name)

                os.system(qsub_call + subprocess_str)

metrics = ['str', 'ac']


# ## Stratified cross-val (nuis, no param optimization)

# In[ ]:


py_script = '/cbica/home/parkesl/research_projects/neurodev_cs_predictive/1_code/cluster/predict_symptoms_scv_nuis.py'
modeldir = outdir+'predict_symptoms_scv_nuis'

for alg in algs:
    for metric in metrics:
        for pheno in phenos:
            for score in scores:
                subprocess_str = '{0} {1} -x {2}X.csv -y {2}y.csv -c {2}c.csv -alg {3} -metric {4} -pheno {5} -score {6} -o {7}'.format(py_exec, py_script, indir, alg, metric, pheno, score, modeldir)

                name = 'null' + '_' + alg + '_' + metric[0] + '_' + pheno[0] + '_' + score[0]
                qsub_call = 'qsub -N {0} -l h_vmem=1G,s_vmem=1G -pe threaded 4 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ '.format(name)

                os.system(qsub_call + subprocess_str)


# ## Random splits cross-val (no nuis, param optimization)

# In[ ]:


py_script = '/cbica/home/parkesl/research_projects/neurodev_cs_predictive/1_code/cluster/predict_symptoms_ncv.py'
modeldir = outdir+'predict_symptoms_ncv'

for alg in algs:
    for metric in metrics:
        for pheno in phenos:
            for score in scores:
                subprocess_str = '{0} {1} -x {2}X.csv -y {2}y.csv -alg {3} -metric {4} -pheno {5} -score {6} -o {7}'.format(py_exec, py_script, indir, alg, metric, pheno, score, modeldir)

                name = 'ncv' + '_' + alg + '_' + metric[0] + '_' + pheno[0] + '_' + score[0]
                qsub_call = 'qsub -N {0} -l h_vmem=1G,s_vmem=1G -t 1-100 -pe threaded 1 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ '.format(name)

                os.system(qsub_call + subprocess_str)


# ## Assemble outputs

# In[ ]:


seeds = np.arange(0,100)
num_seeds = len(seeds)

workdir = outdir+'predict_symptoms_ncv'

nested_score_mean = np.zeros((num_seeds, num_algs, num_scores, num_metrics, num_phenos))
best_alpha = np.zeros((num_seeds, num_algs, num_scores, num_metrics, num_phenos))
run_time_seconds = np.zeros((num_seeds, num_algs, num_scores, num_metrics, num_phenos))

for se, seed in enumerate(seeds):
    print(seed)
    for a, alg in enumerate(algs):
        for s, score in enumerate(scores):
            for m, metric in enumerate(metrics):
                for p, pheno in enumerate(phenos):
                    nested_score = np.loadtxt(os.path.join(workdir, 'split_'+str(seed), alg + '_' + score + '_' + metric + '_' + pheno, 'nested_score.csv'))
                    nested_score_mean[se,a,s,m,p] = nested_score.mean()

                    best_params = json.load(open(os.path.join(workdir, 'split_'+str(seed), alg + '_' + score + '_' + metric + '_' + pheno, 'best_params.json')))
                    best_alpha[se,a,s,m,p] = best_params['reg__alpha']

                    run_time_seconds[se,a,s,m,p] = np.loadtxt(os.path.join(workdir, 'split_'+str(seed), alg + '_' + score + '_' + metric + '_' + pheno, 'run_time_seconds.txt'))

np.save(os.path.join(workdir, 'nested_score_mean'), nested_score_mean)
np.save(os.path.join(workdir, 'best_alpha'), best_alpha)
np.save(os.path.join(workdir, 'run_time_seconds'), run_time_seconds)

