
for i in $(seq $2 1 $3)
do

  nll_weight=$(echo "2-0.1*$i" | bc)
  nll_masked_weight=$(echo "0.1*$i" | bc)

  echo "Training model masked_loss_interpolation_$i with nll weight $nll_weight and nll masked weight $nll_masked_weight"
 
  python train_test.py -c $1 --model_id masked_loss_interpolation_$i --nll_weight $nll_weight --nll_masked_weight $nll_masked_weight --loss_on_masked True --sal_metr_mask True  --big_images False --epochs 50  --compute_ece True  --compute_masked True 

  #Exit if training failed 
  retVal=$?
  if [ $retVal -ne 0 ]; then
    exit $retVal
  fi

done