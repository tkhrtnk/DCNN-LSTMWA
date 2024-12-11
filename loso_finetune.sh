nohup python finetune.py 'configs/esd_finetune_loso0011.json' -w '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0011.json' -a > logs/exp3/esd/loso0011/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/esd_finetune_loso0012.json' -w '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0012.json' -a > logs/exp3/esd/loso0012/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/esd_finetune_loso0013.json' -w '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0013.json' -a > logs/exp3/esd/loso0013/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/esd_finetune_loso0014.json' -w '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0014.json' -a > logs/exp3/esd/loso0014/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/esd_finetune_loso0015.json' -w '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0015.json' -a > logs/exp3/esd/loso0015/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/esd_finetune_loso0016.json' -w '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0016.json' -a > logs/exp3/esd/loso0016/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/esd_finetune_loso0017.json' -w '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0017.json' -a > logs/exp3/esd/loso0017/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/esd_finetune_loso0018.json' -w '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0018.json' -a > logs/exp3/esd/loso0018/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/esd_finetune_loso0019.json' -w '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0019.json' -a > logs/exp3/esd/loso0019/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/esd_finetune_loso0020.json' -w '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0020.json' -a > logs/exp3/esd/loso0020/log_finetune.out &
wait
echo COMPLETED
nohup python calc_avg_result.py 'esd_finetune' >> nohup.out &