nohup python finetune.py 'configs/ravdess_finetune_cv4.json' -w '/work/abelab5/t_tana/emo_clf2/configs/ravdess_train_cv4.json' -a > logs/exp4/ravdess/cv4/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/ravdess_finetune_cv8.json' -w '/work/abelab5/t_tana/emo_clf2/configs/ravdess_train_cv8.json' -a > logs/exp4/ravdess/cv8/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/ravdess_finetune_cv12.json' -w '/work/abelab5/t_tana/emo_clf2/configs/ravdess_train_cv12.json' -a > logs/exp4/ravdess/cv12/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/ravdess_finetune_cv16.json' -w '/work/abelab5/t_tana/emo_clf2/configs/ravdess_train_cv16.json' -a > logs/exp4/ravdess/cv16/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/ravdess_finetune_cv20.json' -w '/work/abelab5/t_tana/emo_clf2/configs/ravdess_train_cv20.json' -a > logs/exp4/ravdess/cv20/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/ravdess_finetune_cv24.json' -w '/work/abelab5/t_tana/emo_clf2/configs/ravdess_train_cv24.json' -a > logs/exp4/ravdess/cv24/log_finetune.out &
wait
echo COMPLETED
nohup python calc_avg_result.py 'ravdess_finetune' >> nohup.out &
