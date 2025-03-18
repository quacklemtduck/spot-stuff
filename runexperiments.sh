cd spot-stuff
python scripts/rsl_rl/train.py --task=Msc-v0 --run_name="main" --headless
git checkout catch_box_move
python scripts/rsl_rl/train.py --task=Msc-v0 --run_name="catch_box_move" --headless
git checkout arm_move
python scripts/rsl_rl/train.py --task=Msc-v0 --run_name="arm_move" --headless
git checkout tanh
python scripts/rsl_rl/train.py --task=Msc-v0 --run_name="tanh" --headless
git checkout good-boy-no-move
python scripts/rsl_rl/train.py --task=Msc-v0 --run_name="goodboy" --headless
git checkout main