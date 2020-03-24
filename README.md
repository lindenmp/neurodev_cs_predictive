# Normative Neurodevelopment: Cross-sectional predictive modeling
<!-- This repository includes code used to analyze the relationship between dimensional psychopathology phenotypes and deviations from normative neurodevelopment in the Philadelphia Neurodevelopmental Cohort. -->

# Environment build

    conda create -n neurodev_cs_predictive python=3.7
    conda activate neurodev_cs_predictive
    # Essentials
    pip install jupyterlab ipython pandas numpy scipy seaborn matplotlib
    pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install
	# Statistics
	pip install statsmodels sklearn tensorflow pingouin
	# Extras
    pip install nibabel torch glob3
    cd /Users/lindenmp/Dropbox/Work/ResProjects/neurodev_cs_predictive
    conda env export > environment.yml
	pip freeze > requirements.txt

# Environment build (cluster)

	# Make environment
	module load python/3.7.3-system
	cd /home/lindenmp/virtual_env
	virtualenv NormativeNeuroDev_CrossSec_DWI
	source /home/lindenmp/virtual_env/NormativeNeuroDev_CrossSec_DWI/bin/activate

	# Essentials
	pip install --upgrade pip
	pip install ipython pandas scipy nibabel sklearn torch glob3

	# Nispat
	mkdir -p /scratch/kg98/Linden/ResProjects/NormativeNeuroDev_CrossSec_DWI/
	cd /scratch/kg98/Linden/ResProjects/NormativeNeuroDev_CrossSec_DWI/
	git clone https://github.com/lindenmp/nispat.git
	cd /scratch/kg98/Linden/ResProjects/NormativeNeuroDev_CrossSec_DWI/nispat
	python setup.py install

	cd /scratch/kg98/Linden/ResProjects/NormativeNeuroDev_CrossSec_DWI/
	pip freeze > requirements_m3.txt


<!-- # Code

In the **code** subdirectory you will find the following Jupyter notebooks and .py scripts:
1. Pre-normative modeling scripts:
- `get_train_test.ipynb`
	- Performs initial ingest of PNC demographic data, participant exclusion based on various quality control.
	- Produces Figures 2A and 2B.
	- Designates train/test split.
- `compute_node_metrics.ipynb`
	- Reads in neuroimaging data.
	- Sets up feature table of regional brain features.
- `clean_node_metrics.ipynb`
	- Performs nuisance regression on feature table.
- `prepare_normative.ipynb`
	- Prepares input files for normative modeling.

2. Run normative modeling:
- `run_normative_local.py`
	- Runs primary normative models on local machine.
- `cluster/run_normative_perm.sh`
	- Submits each of the permuted normative models to the cluster as a separate job

3. Results:
- `results_s1.ipynb`
	- Produces Figure 2C
- `results_s2.ipynb`
	- Produces Figures 3 and 4
- `results_s3.ipynb`
	- Produces Figure 5 -->
