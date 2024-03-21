# Experiment 1.1:
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 2 -P 2 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 4 -P 2 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 8 -P 2 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 16 -P 2 -H 10

srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 2 -P 2 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 4 -P 2 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 8 -P 2 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 16 -P 2 -H 10


# Experiment 1.2:
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 2 -P 32 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 2 -P 64 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 2 -P 128 -H 10

srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 4 -P 32 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 4 -P 64 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 4 -P 128 -H 10

srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 8 -P 32 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 8 -P 64 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 8 -P 128 -H 10

# Experiment 1.3:
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 -L 2 -P 2 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 -L 4 -P 2 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 -L 8 -P 2 -H 10

# Experiment 1.4:
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 8 -P 2 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 16 -P 2 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 32 -P 2 -H 10

srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 256 -L 2 -P 2 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 256 -L 4 -P 2 -H 10
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 128 256 -L 8 -P 2 -H 10
