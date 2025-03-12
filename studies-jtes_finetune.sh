# STUDIESで事前学習させる
nohup python finetune.py 'configs/studies-jtes_finetune_cv1.json' -w '/work/abelab5/t_tana/emo_clf2/configs/studies-jtes_train_cv1.json' -a > logs/exp5/studies-jtes/cv1/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/studies-jtes_finetune_cv2.json' -w '/work/abelab5/t_tana/emo_clf2/configs/studies-jtes_train_cv2.json' -a > logs/exp5/studies-jtes/cv2/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/studies-jtes_finetune_cv3.json' -w '/work/abelab5/t_tana/emo_clf2/configs/studies-jtes_train_cv3.json' -a > logs/exp5/studies-jtes/cv3/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/studies-jtes_finetune_cv4.json' -w '/work/abelab5/t_tana/emo_clf2/configs/studies-jtes_train_cv4.json' -a > logs/exp5/studies-jtes/cv4/log_finetune.out &
wait
echo COMPLETED
nohup python finetune.py 'configs/studies-jtes_finetune_cv5.json' -w '/work/abelab5/t_tana/emo_clf2/configs/studies-jtes_train_cv5.json' -a > logs/exp5/studies-jtes/cv5/log_finetune.out &
wait
echo COMPLETED
nohup python calc_avg_result.py 'studies-jtes_finetune' >> nohup.out &
