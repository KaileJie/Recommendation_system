#!/bin/bash

# Get current timestamp to append to the output filename
timestamp=$(date -u +"%Y%m%d_%H%M%S")

# Define output directory
output_dir="/Users/dallylovely/Desktop/CCGG/Projects/pbs_spider/output"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Run Scrapy spider and save output to a timestamped JSON file
echo "Running spider with output to: $output_dir/pbs_${timestamp}.json"
scrapy crawl pbs_economy -a max_pages=10 -o "$output_dir/pbs_${timestamp}.json"

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Scrapy finished successfully."
else
    echo "❌ Scrapy encountered an error."
fi
