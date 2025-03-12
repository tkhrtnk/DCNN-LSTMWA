nohup python finetune.py 'configs/jtes_finetune_all.json' -w '/work/abelab5/t_tana/emo_clf2/configs/jtes_train_all.json' -a > logs/exp3/jtes/all/log_finetune.out &
wait
echo COMPLETED