cd spot-stuff
git checkout fixed-body
python scripts/rsl_rl/train.py --task=Msc-v0 --run_name="fixed-body" --headless
git checkout fixed-body-gravity
python scripts/rsl_rl/train.py --task=Msc-v0 --run_name="fixed-body-gravity" --headless
git checkout fixed-body