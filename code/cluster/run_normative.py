#import nispat
import os, sys
from nispat.normative import estimate
from nispat.normative_parallel import execute_nm, collect_nm, delete_nm

# ------------------------------------------------------------------------------
# parallel (batch)
# ------------------------------------------------------------------------------
# settings and paths
python_path = '/home/lindenmp/virtual_env/nispat/bin/python'
normative_path = '/home/lindenmp/virtual_env/nispat/nispat/nispat/normative.py'
batch_size = 16
memory = '2G'
duration = '2:00:00'
cluster_spec = 'm3'

# ------------------------------------------------------------------------------
# Normative dir
# ------------------------------------------------------------------------------
# primary directory for normative model
train_test_str = 'squeakycleanExclude'
exclude_str = 't1Exclude'
combo_label = 'schaefer_200_streamlineCount_consist'

normativedir = os.path.join('/scratch/kg98/lindenmp/ResProjects/neurodev_cs_predictive/analysis/normative',
	exclude_str, train_test_str, combo_label, 'ageAtScan1_Years+sex_adj/')
print(normativedir)

# ------------------------------------------------------------------------------
# Primary model
job_name = 'prim_'
wdir = normativedir; os.chdir(wdir)

# input files and paths
cov_train = os.path.join(normativedir, 'cov_train.txt')
resp_train = os.path.join(normativedir, 'resp_train.txt')
cov_test = os.path.join(normativedir, 'cov_test.txt')
resp_test = os.path.join(normativedir, 'resp_test.txt')

# run normative
execute_nm(wdir, python_path=python_path, normative_path=normative_path, job_name=job_name, covfile_path=cov_train, respfile_path=resp_train,
           batch_size=batch_size, memory=memory, duration=duration, cluster_spec=cluster_spec, cv_folds=None, testcovfile_path=cov_test, testrespfile_path=resp_test, alg = 'gpr')
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Forward model
job_name = 'fwd_'
wdir = os.path.join(normativedir, 'forward/'); os.chdir(wdir)

# input files and paths
cov_train = os.path.join(normativedir, 'cov_train.txt');
resp_train = os.path.join(normativedir, 'resp_train.txt');
cov_test = os.path.join(wdir, 'synth_cov_test.txt');

# run normative
execute_nm(wdir, python_path=python_path, normative_path=normative_path, job_name=job_name, covfile_path=cov_train, respfile_path=resp_train,
           batch_size=batch_size, memory=memory, duration=duration, cluster_spec=cluster_spec, cv_folds=None, testcovfile_path=cov_test, testrespfile_path=None, alg = 'gpr')
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
wdir = normativedir; os.chdir(wdir)
wdir = os.path.join(normativedir, 'forward/'); os.chdir(wdir)

collect_nm(wdir, collect=True)
delete_nm(wdir)

# ------------------------------------------------------------------------------
