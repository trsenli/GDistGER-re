#
# Script Name: run.sh
# Description: The run shell script of GDistGER.
# Author: lzl
# Date: 2024/11/19
#
# Usage:
#  ./your_script_name.sh [option1] [option2] ...
#  ./run.sh <graph> 
#
# Options:
#  graph, the graph dataset.
#
# Example Usage:
#  ./run.sh wiki
#
# Notes:
#  Testing for GDistGER, Args includes sampling args, learning args, distribute args and openmpi args.
#

#sync_size_cal=$(echo "2^$2" | bc)

export OMPI_ALLOW_RUN_AS_ROOT=1 
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 
sync_size_cal=1024

BIN=./bin/huge_walk
GRAPH_PREFIX=/home/lzl/nfs.d/dataset/original_bin
TRAIN_GRAPH=../dataset/train_bin/$1_train.data
NODE_NUM=1

DISTRIBUTE_ARGS="-hostfile ./hosts "
OPENMPI_CONFIG_ARGS=" -mca btl ^openib -mca mtl ^ofi -mca btl_tcp_if_include ens17f0 "
GRAPH_NAME=$1
GRAPH=$GRAPH_PREFIX/${GRAPH_NAME}.data

if [ $GRAPH_NAME = "wiki" ]; then
	V_NUM=7115 W_NUM=$V_NUM MIN_L=20 MIN_R=1	
elif [ $GRAPH_NAME = "ytb" ]; then
	V_NUM=1138499  W_NUM=$V_NUM MIN_L=20 MIN_R=10
elif [ $GRAPH_NAME = "soc" ]; then
	V_NUM=1632803  W_NUM=$V_NUM MIN_L=20 MIN_R=5
elif [ $GRAPH_NAME = "LJ" ]; then
	V_NUM=2238731 W_NUM=$V_NUM MIN_L=20 MIN_R=5
elif [ $GRAPH_NAME = "com" ]; then
	V_NUM=3072441  W_NUM=$V_NUM MIN_L=20 MIN_R=5 
elif [ $GRAPH_NAME = "twt" ]; then
	V_NUM=41652230  W_NUM=$V_NUM  MIN_L=20 MIN_R=5
fi

SAMPLE_ARGS="-g $GRAPH \
	-v $V_NUM \
	-w $W_NUM \
	--min_L $MIN_L \
	--min_R $MIN_R \
	-o ./out/$GRAPH_NAME \
	--make-undirected"

LEARNING_ARGS="-emb_output ./out/${GRAPH_NAME}_emb.txt \
	 	-size 100 \
		-iter 3 \
		-threads 10 \
		-window 10 \
		-negative 5 \
		-batch-size 21 \
		-min-count 0 \
		-sample 1e-3 \
		-alpha 0.01 \
		-debug 2 \
		-sync-size $sync_size_cal"

mpirun $OPENMPI_CONFIG_ARGS \
	$DISTRIBUTE_ARGS \
	-npernode $NODE_NUM \
	$BIN \
	$SAMPLE_ARGS \
	$LEARNING_ARGS 
