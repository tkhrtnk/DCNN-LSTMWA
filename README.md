# step0: prepare dataset
python filelists/[dname]/\*\*/011_prep_\*\*.py

# step1: finetuning
nohup ./**_finetune.sh > nohup.out &

# step2: train
nohup ./**_train.sh > nohup2.out &

# step3: inference
python inference.py