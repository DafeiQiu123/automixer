module load gcc/6.5.0-lwshmxc
WORKING_DIR=$(pwd)
MAP_PROJ_DIR=$(pwd)/MERT

source .venv/bin/activate
export PATH=${WORKING_DIR}/cuda-11.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${WORKING_DIR}/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=${WORKING_DIR}/cuda-11.3
echo export PATH=${WORKING_DIR}/cuda-11.3/bin${PATH:+:${PATH}} >> ~/.bashrc
echo export LD_LIBRARY_PATH=${WORKING_DIR}/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} >> ~/.bashrc
echo export CUDA_HOME=${WORKING_DIR}/cuda-11.3 >> ~/.bashrc
nvcc -V
