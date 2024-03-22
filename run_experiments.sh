# Experiment 1.1:
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 2 -P 4 -H 100 -n "exp1_1" --checkpoints 2
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 4 -P 4 -H 100 -n "exp1_1" --checkpoints 2 
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 8 -P 4 -H 100 -n "exp1_1" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 16 -P 4 -H 100 -n "exp1_1" --checkpoints 2  

srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 2 -P 4 -H 100 -n "exp1_1" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 4 -P 4 -H 100 -n "exp1_1" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 8 -P 4 -H 100 -n "exp1_1" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 16 -P 4 -H 100 -n "exp1_1" --checkpoints 2  


# Experiment 1.2:
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 2 -P 3 -H 100 -n "exp1_2" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 2 -P 3 -H 100 -n "exp1_2" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 128 -L 2 -P 3 -H 100 -n "exp1_2" --checkpoints 2  

srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 4 -P 3 -H 100 -n "exp1_2" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 4 -P 3 -H 100 -n "exp1_2" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 128 -L 4 -P 3 -H 100 -n "exp1_2" --checkpoints 2  

srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 8 -P 3 -H 100 -n "exp1_2" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 8 -P 3 -H 100 -n "exp1_2" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 128 -L 8 -P 3 -H 100 -n "exp1_2" --checkpoints 2  


# Experiment 1.3:
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 -L 2 -P 3 -H 70 -n "exp1_3" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 -L 3 -P 3 -H 70 -n "exp1_3" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 -L 4 -P 3 -H 70 -n "exp1_3" --checkpoints 2  


# Experiment 1.4:
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 8 -P 8 -H 50 -n "exp1_4" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 16 -P 8 -H 50 -n "exp1_4" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 32 -P 8 -H 50 -n "exp1_4" --checkpoints 2  

srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 256 -L 2 -P 8 -H 50 -n "exp1_4" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 256 -L 4 -P 8 -H 50 -n "exp1_4" --checkpoints 2  
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 256 -L 8 -P 8 -H 50 -n "exp1_4" --checkpoints 2  
