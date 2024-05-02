Please refer to https://github.com/karpathy/nanoGPT for the original repo
This repo contains unassociated personal experiments

python data/shakespeare_char/scratch.py

python data/shakespeare_char/prepare.py

python train.py config/train_shakespeare_char.py 
--block_size=32

python sample.py --out_dir=out-shakespeare-char --start="ABC" --num_samples=10 --max_new_tokens=2 --device=cpu