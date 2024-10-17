HF_ENDPOINT=https://hf-mirror.com python3 -u train.py --gpu_id 1 --seed 3407 --batch_size 8 --dataset uhdiqa --loss MSE --model MobileVit_IQA --save_path ./Running_Test

HF_ENDPOINT=https://hf-mirror.com python3 -u train.py --gpu_id 1 --seed 3407 --batch_size 8 --dataset uhdiqa --loss MSE --save_path ./Running_Distill --teacher_pkl YOUR_TEACHER_PKL
