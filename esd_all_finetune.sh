nohup python finetune.py 'configs/esd_finetune_all.json' -w '/work/abelab5/t_tana/emo_clf2/configs/esd_train_all.json' -a > logs/exp3/esd/all/log_finetune.out &
wait
echo COMPLETED