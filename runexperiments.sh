cd spot-stuff
python scripts/rsl_rl/train.py --task=Msc-v0 --run_name="main"
git checkout catch_box_move
python scripts/rsl_rl/train.py --task=Msc-v0 --run_name="catch_box_move"
git checkout arm_move
python scripts/rsl_rl/train.py --task=Msc-v0 --run_name="arm_move"
git checkout tanh
python scripts/rsl_rl/train.py --task=Msc-v0 --run_name="tanh"
git checkout good-boy-no-move
python scripts/rsl_rl/train.py --task=Msc-v0 --run_name="goodboy"
git checkout main