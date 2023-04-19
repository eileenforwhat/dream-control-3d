# Run scripts for baseline (--guidance stable-diffusion) results.
DATA_DIR=/mnt/data

python main.py --text "a wooden chair with cushions on the seat" --workspace $DATA_DIR/baseline_chair -O
python main.py --workspace $DATA_DIR/baseline_chair -O --test

python main.py --text "a black horse" --workspace $DATA_DIR/baseline_horse -O
python main.py --workspace $DATA_DIR/baseline_horse -O --test

python main.py --text "a black and white zebra" --workspace $DATA_DIR/baseline_zebra -O
python main.py --workspace $DATA_DIR/baseline_zebra -O --test

python main.py --text "a hot air balloon made of wood" --workspace $DATA_DIR/baseline_balloon_1 -O
python main.py --workspace $DATA_DIR/baseline_balloon_1 -O --test

python main.py --text "a hot air balloon made of glass" --workspace $DATA_DIR/baseline_balloon_2 -O
python main.py --workspace $DATA_DIR/baseline_balloon_2 -O --test

python main.py --text "Bird with blue feathers and long wings" --workspace $DATA_DIR/baseline_bird_1 -O
python main.py --workspace $DATA_DIR/baseline_bird_1 -O --test

python main.py --text "Bird with rainbow feathers and long wings" --workspace $DATA_DIR/baseline_bird_2 -O
python main.py --workspace $DATA_DIR/baseline_bird_2 -O --test
