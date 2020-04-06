# CUBIC
# --------------------------------------------------------------------------------
# push data to server
# --------------------------------------------------------------------------------
dir=neurodev_cs_predictive/analysis_cubic/normative/t1Exclude/squeakycleanExclude/schaefer_200_streamlineCount_consist/ageAtScan1_Years+sex_adj
local_dir=/Users/lindenmp/Dropbox/Work/ResProjects/${dir}/*
server_dir=/cbica/home/parkesl/ResProjects/${dir}

ssh cbica "mkdir -p ${server_dir}"

scp -r ${local_dir} parkesl@cbica-cluster.uphs.upenn.edu:${server_dir}

# --------------------------------------------------------------------------------
# pull data from server
# --------------------------------------------------------------------------------
dir=neurodev_cs_predictive/analysis_cubic/normative/t1Exclude/squeakycleanExclude/schaefer_200_streamlineCount_consist/ageAtScan1_Years+sex_adj
local_dir=/Users/lindenmp/Dropbox/Work/ResProjects/${dir}
server_dir=/cbica/home/parkesl/ResProjects/${dir}

# primary files
for file in pRho.txt Rho.txt Z.txt yhat.txt rmse.txt ys2.txt expv.txt smse.txt Hyp_1.txt msll.txt; do scp parkesl@cbica-cluster.uphs.upenn.edu:${server_dir}/${file} ${local_dir}/; done

# forward
for file in yhat.txt ys2.txt; do scp parkesl@cbica-cluster.uphs.upenn.edu:${server_dir}/forward/${file} ${local_dir}/forward/; done
