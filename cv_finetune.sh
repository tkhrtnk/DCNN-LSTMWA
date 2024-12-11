nohup python finetune.py 'configs/jtes_finetune_cv10.json' -w '/work/abelab5/t_tana/emo_clf2/configs/jtes_train_cv10.json' -a > logs/exp3/jtes/cv10/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/jtes_finetune_cv20.json' -w '/work/abelab5/t_tana/emo_clf2/configs/jtes_train_cv20.json' -a > logs/exp3/jtes/cv20/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/jtes_finetune_cv30.json' -w '/work/abelab5/t_tana/emo_clf2/configs/jtes_train_cv30.json' -a > logs/exp3/jtes/cv30/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/jtes_finetune_cv40.json' -w '/work/abelab5/t_tana/emo_clf2/configs/jtes_train_cv40.json' -a > logs/exp3/jtes/cv40/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/jtes_finetune_cv50.json' -w '/work/abelab5/t_tana/emo_clf2/configs/jtes_train_cv50.json' -a > logs/exp3/jtes/cv50/log_finetune.out &
wait
echo COMPLETED
nohup python calc_avg_result.py 'jtes_finetune' >> nohup.out &
