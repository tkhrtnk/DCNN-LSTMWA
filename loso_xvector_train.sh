nohup python train_xvector.py 'configs/esd_xvector_train_loso0011.json' > logs/xvector/esd/loso0011/log_train.out &
wait
echo COMPLETED
nohup python train_xvector.py 'configs/esd_xvector_train_loso0012.json' > logs/xvector/esd/loso0012/log_train.out &
wait
echo COMPLETED
nohup python train_xvector.py 'configs/esd_xvector_train_loso0013.json' > logs/xvector/esd/loso0013/log_train.out &
wait
echo COMPLETED
nohup python train_xvector.py 'configs/esd_xvector_train_loso0014.json' > logs/xvector/esd/loso0014/log_train.out &
wait
echo COMPLETED
nohup python train_xvector.py 'configs/esd_xvector_train_loso0015.json' > logs/xvector/esd/loso0015/log_train.out &
wait
echo COMPLETED
nohup python train_xvector.py 'configs/esd_xvector_train_loso0016.json' > logs/xvector/esd/loso0016/log_train.out &
wait
echo COMPLETED
nohup python train_xvector.py 'configs/esd_xvector_train_loso0017.json' > logs/xvector/esd/loso0017/log_train.out &
wait
echo COMPLETED
nohup python train_xvector.py 'configs/esd_xvector_train_loso0018.json' > logs/xvector/esd/loso0018/log_train.out &
wait
echo COMPLETED
nohup python train_xvector.py 'configs/esd_xvector_train_loso0019.json' > logs/xvector/esd/loso0019/log_train.out &
wait
echo COMPLETED
nohup python train_xvector.py 'configs/esd_xvector_train_loso0020.json' > logs/xvector/esd/loso0020/log_train.out &
wait
echo COMPLETED
nohup python calc_avg_result.py 'esd_xvector_train' >> nohup.out &
