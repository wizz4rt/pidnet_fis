import subprocess

subprocess.call(['python', 'tools/train.py', '--cfg', 'configs/facial/class_ablation/0005-24-adamw-noleftright.yaml'], check=True)
subprocess.call(['python', 'tools/train.py', '--cfg', 'configs/facial/class_ablation/0005-24-adamw-noneck.yaml'], check= True)
subprocess.call(['python', 'tools/train.py', '--cfg', 'configs/facial/model_size/0005-24-adamw-M.yaml'], check=True)