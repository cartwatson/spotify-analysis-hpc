#!/bin/bash

export TESTING=true

./scripts/build_and_run.sh
mv src/data/output.csv src/data/output1.csv

./scripts/build_and_run.sh
mv src/data/output.csv src/data/output2.csv

git --no-pager diff --no-index src/data/output1.csv src/data/output2.csv > /dev/null
if [ $? -eq 0 ]
then
    echo
    echo "TEST PASSED"
else
    echo 
    echo "DIFFERENCES FOUND"
    git --no-pager diff --no-index --color src/data/output1.csv src/data/output2.csv | sed -n '3,20p'
    echo "..."
    echo
    echo "TEST FAILED"
    echo
fi

rm src/data/output1.csv src/data/output2.csv
unset TESTING