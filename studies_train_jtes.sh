# JTESで事前学習済みのモデルに対してSTUDIES-teacherで再学習（STUDIESでファインチューニングする）
nohup python train.py 'configs/studies_train_jtes_cv20.json' > logs/exp5/studies/cv20/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/studies_train_jtes_cv40.json' > logs/exp5/studies/cv40/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/studies_train_jtes_cv60.json' > logs/exp5/studies/cv60/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/studies_train_jtes_cv80.json' > logs/exp5/studies/cv80/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/studies_train_jtes_cv100.json' > logs/exp5/studies/cv100/log_train.out &
wait
echo COMPLETED
nohup python calc_avg_result.py 'studies_train_jtes' >> nohup.out &
