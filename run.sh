for model in 1 2 3
do  
    echo "Running model $model"
    python inference.py --model $model
done
