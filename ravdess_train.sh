nohup python train.py 'configs/ravdess_train_cv4.json' > logs/exp4/ravdess/cv4/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/ravdess_train_cv8.json' > logs/exp4/ravdess/cv8/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/ravdess_train_cv12.json' > logs/exp4/ravdess/cv12/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/ravdess_train_cv16.json' > logs/exp4/ravdess/cv16/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/ravdess_train_cv20.json' > logs/exp4/ravdess/cv20/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/ravdess_train_cv24.json' > logs/exp4/ravdess/cv24/log_train.out &
wait
echo COMPLETED
nohup python calc_avg_result.py 'ravdess_train' >> nohup.out &