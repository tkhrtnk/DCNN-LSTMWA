# STUDIESで事前学習させる
nohup python finetune.py 'configs/studies_finetune_cv20.json' -w '/work/abelab5/t_tana/emo_clf2/configs/studies_train_ita_cv20.json' -a > logs/exp4/studies/cv20/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/studies_finetune_cv40.json' -w '/work/abelab5/t_tana/emo_clf2/configs/studies_train_ita_cv40.json' -a > logs/exp4/studies/cv40/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/studies_finetune_cv60.json' -w '/work/abelab5/t_tana/emo_clf2/configs/studies_train_ita_cv60.json' -a > logs/exp4/studies/cv60/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/studies_finetune_cv80.json' -w '/work/abelab5/t_tana/emo_clf2/configs/studies_train_ita_cv80.json' -a > logs/exp4/studies/cv80/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/studies_finetune_cv100.json' -w '/work/abelab5/t_tana/emo_clf2/configs/studies_train_ita_cv100.json' -a > logs/exp4/studies/cv100/log_finetune.out &
wait
echo COMPLETED
nohup python calc_avg_result.py 'studies_finetune' >> nohup.out &
