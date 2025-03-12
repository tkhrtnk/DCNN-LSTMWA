nohup python train.py 'configs/esd_train_loso0011.json' > logs/exp3/esd/loso0011/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/esd_train_loso0012.json' > logs/exp3/esd/loso0012/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/esd_train_loso0013.json' > logs/exp3/esd/loso0013/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/esd_train_loso0014.json' > logs/exp3/esd/loso0014/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/esd_train_loso0015.json' > logs/exp3/esd/loso0015/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/esd_train_loso0016.json' > logs/exp3/esd/loso0016/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/esd_train_loso0017.json' > logs/exp3/esd/loso0017/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/esd_train_loso0018.json' > logs/exp3/esd/loso0018/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/esd_train_loso0019.json' > logs/exp3/esd/loso0019/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/esd_train_loso0020.json' > logs/exp3/esd/loso0020/log_train.out &
wait
echo COMPLETED
nohup python calc_avg_result.py 'esd_train' >> nohup.out &
