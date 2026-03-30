# Function-Based RL Scripts

คำสั่งที่ใช้บ่อยสำหรับแต่ละ algorithm

## Linear_Q

Train GUI
```bash
python CartPole_4.5.0/scripts/Function_based/Linear_Q/train.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0
```

Train headless
```bash
python CartPole_4.5.0/scripts/Function_based/Linear_Q/train.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

Play headless
```bash
python CartPole_4.5.0/scripts/Function_based/Linear_Q/play.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

TensorBoard
```bash
tensorboard --logdir CartPole_4.5.0/runs/Function_based/Linear_Q
```

## DQN

Train headless
```bash
python CartPole_4.5.0/scripts/Function_based/DQN/train.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

Play headless
```bash
python CartPole_4.5.0/scripts/Function_based/DQN/play.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

TensorBoard
```bash
tensorboard --logdir CartPole_4.5.0/runs/Function_based/DQN
```

## MC_REINFORCE

Train headless
```bash
python CartPole_4.5.0/scripts/Function_based/MC_REINFORCE/train.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

Play headless
```bash
python CartPole_4.5.0/scripts/Function_based/MC_REINFORCE/play.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

TensorBoard
```bash
tensorboard --logdir CartPole_4.5.0/runs/Function_based/MC_REINFORCE
```

## AC

Train headless
```bash
python CartPole_4.5.0/scripts/Function_based/AC/train.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

Play headless
```bash
python CartPole_4.5.0/scripts/Function_based/AC/play.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

TensorBoard
```bash
tensorboard --logdir CartPole_4.5.0/runs/Function_based/AC
```

## A2C

Train headless
```bash
python CartPole_4.5.0/scripts/Function_based/A2C/train.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

Play headless
```bash
python CartPole_4.5.0/scripts/Function_based/A2C/play.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

TensorBoard
```bash
tensorboard --logdir CartPole_4.5.0/runs/Function_based/A2C
```

## PPO

Train headless
```bash
python CartPole_4.5.0/scripts/Function_based/PPO/train.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

Play headless
```bash
python CartPole_4.5.0/scripts/Function_based/PPO/play.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

TensorBoard
```bash
tensorboard --logdir CartPole_4.5.0/runs/Function_based/PPO
```

## TD3

Train headless
```bash
python CartPole_4.5.0/scripts/Function_based/TD3/train.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

Play headless
```bash
python CartPole_4.5.0/scripts/Function_based/TD3/play.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

TensorBoard
```bash
tensorboard --logdir CartPole_4.5.0/runs/Function_based/TD3
```

## SAC

Train headless
```bash
python CartPole_4.5.0/scripts/Function_based/SAC/train.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

Play headless
```bash
python CartPole_4.5.0/scripts/Function_based/SAC/play.py --task Stabilize-Isaac-Cartpole-v0 --device cuda:0 --headless
```

TensorBoard
```bash
tensorboard --logdir CartPole_4.5.0/runs/Function_based/SAC
```
