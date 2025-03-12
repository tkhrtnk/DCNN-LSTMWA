# STUDIESで事前学習済みのモデルに対してSTUDIES-teacherで再学習（STUDIESでファインチューニングする）
# studies_finetuneからの実行
nohup python train.py 'configs/studies_train_ita_cv20.json' > logs/exp4/studies/cv20/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/studies_train_ita_cv40.json' > logs/exp4/studies/cv40/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/studies_train_ita_cv60.json' > logs/exp4/studies/cv60/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/studies_train_ita_cv80.json' > logs/exp4/studies/cv80/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/studies_train_ita_cv100.json' > logs/exp4/studies/cv100/log_train.out &
wait
echo COMPLETED
nohup python calc_avg_result.py 'studies_train_ita' >> nohup.out &
