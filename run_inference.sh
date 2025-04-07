#!/bin/bash

echo "=================================================="
echo "          GHOST 2.0 Inference Runner             "
echo "=================================================="
echo ""

start_time=$(date +%s)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Activating conda environment 'ghost-new'..."
source ~/miniconda/bin/activate ghost-new
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Environment activated successfully"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running inference script..."
echo "=================================================="

# Run the inference script
python inference.py --source ./examples/images/hab.jpg --target ./examples/images/elon.jpg --save_path result.png

status=$?

echo ""
echo "=================================================="
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

if [ $status -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Inference completed successfully!"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Result saved to: $(pwd)/result.png"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total execution time: ${minutes}m ${seconds}s"
    echo ""
    echo "To view the result, open result.png"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Inference failed with error code: $status"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total execution time: ${minutes}m ${seconds}s"
fi

echo "=================================================="
