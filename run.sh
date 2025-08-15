
nohup python3 -u main_mnist.py --dataset mnist --alpha 0.1 --n_users 20 --k 7 --rounds 600 --batch_size 64 --n_batches -1 --surrogate --lr 0.001 --test_bs 2048 --step_size 0.3 --n_intervals -1 --steps -1 --gpu_ids 0 > train_log.log


nohup python3 -u main_mnist.py --dataset mnist --alpha 0.5 --n_users 20 --k 7 --rounds 150 --batch_size 64 --n_batches -1 --surrogate --lr 0.001 --test_bs 2048 --step_size 0.3 --n_intervals -1 --steps -1 --gpu_ids 0 > train_log.log


