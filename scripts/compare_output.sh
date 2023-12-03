#!/bin/bash

export TESTING=true

./scripts/build_and_run.sh
mv src/data/output.csv src/data/output1.csv

./scripts/build_and_run.sh
mv src/data/output.csv src/data/output2.csv

diff src/data/output1.csv src/data/output2.csv > /dev/null
if [ $? -eq 0 ]; then
    echo "TEST PASSED"
else
    echo "TEST FAILED"
fi

rm src/data/output1.csv src/data/output2.csv

unset TESTING