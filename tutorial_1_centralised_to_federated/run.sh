#!/bin/bash
NUM_CLIENTS=2

echo "Starting server"
python server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

echo "Create partitions"
python 
echo $NUM_CLIENTS
for i in {1..$NUM_CLIENTS}
do
    echo "Starting client $i"
    python client.py --cid $i &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
