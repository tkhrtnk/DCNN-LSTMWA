nohup python train.py 'configs/jtes_train_cv10.json' > logs/exp3/jtes/cv10/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/jtes_train_cv20.json' > logs/exp3/jtes/cv20/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/jtes_train_cv30.json' > logs/exp3/jtes/cv30/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/jtes_train_cv40.json' > logs/exp3/jtes/cv40/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/jtes_train_cv50.json' > logs/exp3/jtes/cv50/log_train.out &
wait
echo COMPLETED
nohup python calc_avg_result.py 'jtes_train' >> nohup.out &
