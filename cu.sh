#!/bin/bash
echo "welcome"

ls | grep -P "slurm.*" | xargs -d"\n" rm
find . -name 'job_*' | tar -czvf dump.tar.gz --files-from -
ls | grep -P 'job_' | xargs -d"\n" rm

echo "Deleted all the slurm files"
echo "Tarred all the pickles"
echo "Deleted all the pickles"
