python3 train_test.py --config_paths $1 $2

retVal=$?
if [ $retVal -ne 0 ]; then
    exit
fi

python3 compute_and_evaluate_explanations.py --config_paths $1 $2

retVal=$?
if [ $retVal -ne 0 ]; then
    exit
fi

python3 krippendorf_alpha.py $2

retVal=$?
if [ $retVal -ne 0 ]; then
    exit
fi

python3 test_size_reduction.py --multi_processes_mode $2