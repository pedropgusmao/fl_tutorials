#!/bin/bash

# To make sure that relative paths work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

NUM_CLIENTS=2
ALPHA=100000

SERVER_LR=1.0 # <--- NEW

####
ALPHA_TEMP=$(printf %.2f $ALPHA)
PARTITIONS_DIR="dataset/lda/${NUM_CLIENTS}/${ALPHA_TEMP}/"
echo "Starting server"
python server.py --num_clients $NUM_CLIENTS --server_learning_rate $SERVER_LR & # <--- NEW
sleep 3  # Sleep for 3s to give the server enough time to start

echo "Create partitions"
python ../../tutorial_1_centralised_to_federated/solution/create_partitions.py

for (( i=0; i<$NUM_CLIENTS; i++ ))
do
    echo "Starting client $i"
    python ../../tutorial_1_centralised_to_federated/solution/client.py --cid $i --partitions_root $PARTITIONS_DIR &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait