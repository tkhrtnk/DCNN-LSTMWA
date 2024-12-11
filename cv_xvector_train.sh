nohup python train_xvector.py 'configs/jtes_xvector_train_cv10.json' > logs/xvector/jtes/cv10/log_train.out &
wait
echo COMPLETED
nohup python train_xvector.py 'configs/jtes_xvector_train_cv20.json' > logs/xvector/jtes/cv20/log_train.out &
wait
echo COMPLETED
nohup python train_xvector.py 'configs/jtes_xvector_train_cv30.json' > logs/xvector/jtes/cv30/log_train.out &
wait
echo COMPLETED
nohup python train_xvector.py 'configs/jtes_xvector_train_cv40.json' > logs/xvector/jtes/cv40/log_train.out &
wait
echo COMPLETED
nohup python train_xvector.py 'configs/jtes_xvector_train_cv50.json' > logs/xvector/jtes/cv50/log_train.out &
wait
echo COMPLETED
nohup python calc_avg_result.py 'jtes_xvector_train' >> nohup.out &
