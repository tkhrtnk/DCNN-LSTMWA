# STUDIESで事前学習済みのモデルに対してSTUDIES-teacherで再学習（STUDIESでファインチューニングする）
# studies-jtes_finetuneからの実行
nohup python train.py 'configs/studies-jtes_train_cv1.json' > logs/exp5/studies-jtes/cv1/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/studies-jtes_train_cv2.json' > logs/exp5/studies-jtes/cv2/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/studies-jtes_train_cv3.json' > logs/exp5/studies-jtes/cv3/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/studies-jtes_train_cv4.json' > logs/exp5/studies-jtes/cv4/log_train.out &
wait
echo COMPLETED
nohup python train.py 'configs/studies-jtes_train_cv5.json' > logs/exp5/studies-jtes/cv5/log_train.out &
wait
echo COMPLETED
nohup python calc_avg_result.py 'studies-jtes_train' >> nohup.out &
