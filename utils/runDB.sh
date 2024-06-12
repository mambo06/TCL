rm -r output/$1/mlp/tuning/reproduced
rm -r output/$1/mlp/tuned_reproduced
mkdir data/adult/normal
mkdir data/adult/ood

# you can choose any other name instead of "reproduced.toml"; it is better to keep this
# name while completing the tutorial
cp output/$1/mlp/tuning/0.toml output/$1/mlp/tuning/reproduced.toml
# let's reduce the number of tuning iterations to make tuning fast (and ineffective)
python -c "
from pathlib import Path
p = Path('output/$1/mlp/tuning/reproduced.toml')
p.write_text(p.read_text().replace('n_trials = 100', 'n_trials = 5'))
"
python bin/tune.py output/$1/mlp/tuning/reproduced.toml

# Evaluation
# create a directory for evaluation
mkdir -p output/$1/mlp/tuned_reproduced

# clone the best config from the tuning stage with 15 different random seeds
python -c "
for seed in range(15):
    open(f'output/$1/mlp/tuned_reproduced/{seed}.toml', 'w').write(
        open('output/$1/mlp/tuning/reproduced/best.toml').read().replace('seed = 0', f'seed = {seed}')
    )
"

# train MLP with all 15 configs
for seed in {0..14}
do
    python bin/mlp.py output/$1/mlp/tuned_reproduced/${seed}.toml
done