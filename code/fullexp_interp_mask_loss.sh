expl_groups=("all")

if test "$( nvidia-smi | grep "A40" | wc -l)" -gt 0; then
    val_batch_size=1600
    expl_batch_size=200
elif [ "$CUDA_VISIBLE_DEVICES" == "0" ] || [ -d "/scratch" ]; then
    val_batch_size=200
    expl_batch_size=90
else
    val_batch_size=800
    expl_batch_size=100
fi

val_batch_size_arg="--val_batch_size "${val_batch_size}

if [ -d "/scratch" ]; then
    num_workers_arg="--num_workers 2"
else
    num_workers_arg=""
fi


for i in $(seq $2 1 $3)
do

    nll_weight=$(echo "2-0.1*$i" | bc)
    nll_masked_weight=$(echo "0.1*$i" | bc)

    model=masked_loss_interpolation_$i

    #Set arg $1 to upper case
    config=${1^^}
    #Remove model_ prefix
    config=${config#MODEL_}
    #Remove 26.config suffix
    exp_id=${config%.CONFIG}
    echo EXP_ID: $exp_id

    echo $model $val_batch_size_arg

    echo "Training model $model with nll weight $nll_weight and nll masked weight $nll_masked_weight"

    python train_test.py -c $1 --model_id $model --nll_weight $nll_weight --nll_masked_weight $nll_masked_weight --loss_on_masked True --sal_metr_mask True  --big_images False --epochs 50  --compute_ece True  --compute_masked True

    retVal=$?
    if [ $retVal -ne 0 ]; then
        exit
    fi

    #./eval_expl_25.sh $1 all all $model all default $expl_batch_size
    if [ ! -f ../results/${exp_id}/krippendorff_alpha_values_list_${model}_bNone_all.csv ]; then
        ./eval_expl_25.sh $1 all scorecam $model all default 200
    fi 

    retVal=$?
    if [ $retVal -ne 0 ]; then
        exit
    fi

    for expl_group in "${expl_groups[@]}"
    do
        echo $expl_group

        if [ ! -f ../results/${exp_id}/krippendorff_alpha_values_list_${model}_bNone_{expl_group}.csv ]; then

            python3 does_cumulative_increase_interrater_reliability.py -c $1 --model_id $model --ordinal_metric --post_hoc_group $expl_group

            retVal=$?
            if [ $retVal -ne 0 ]; then
                exit
            fi

        fi

    done
done

python3 krippendorf_vs_interp.py -c $1 --model_id_prefix masked_loss_interpolation_ --model_id_start noneRed2_lr --model_id_end noneRed_onlylossonmasked_multiobj2_lr 