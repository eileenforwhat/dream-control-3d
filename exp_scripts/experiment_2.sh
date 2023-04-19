# Run scripts for exp2 (--guidance controlnet) results, with extra training steps for single default view (based on known_view_interval)
DATA_DIR=/mnt/data

python main.py --text "a wooden chair with cushions on the seat" --workspace $DATA_DIR/exp2_chair -O --guidance controlnet --guidance_image_path scribbles/chair.png --known_view_interval 5
python main.py --workspace $DATA_DIR/exp2_chair -O --test

python main.py --text "a wooden chair with cushions on the seat" --workspace $DATA_DIR/exp2_chair_2 -O --guidance controlnet --guidance_image_path scribbles/chair_2.png --known_view_interval 2
python main.py --workspace $DATA_DIR/exp2_chair_2 -O --test

python main.py --text "a wooden chair with cushions on the seat" --workspace $DATA_DIR/exp2_couch -O --guidance controlnet --guidance_image_path scribbles/couch.png --known_view_interval 5
python main.py --workspace $DATA_DIR/exp2_couch -O --test

python main.py --text "a black horse" --workspace $DATA_DIR/exp2_horse -O --guidance controlnet --guidance_image_path scribbles/horse.png --known_view_interval 5
python main.py --workspace $DATA_DIR/exp2_horse -O --test

python main.py --text "a black and white zebra" --workspace $DATA_DIR/exp2_zebra -O --guidanace controlnet --guidance_image_path scribbles/horse.png --known_view_interval 5
python main.py --workspace $DATA_DIR/exp2_zebra -O --test

python main.py --text "a hot air balloon made of wood" --workspace $DATA_DIR/exp2_balloon_1 -O --guidance controlnet --guidance_image_path scribbles/balloon.png --known_view_interval 5
python main.py --workspace $DATA_DIR/exp2_balloon_1 -O --test

python main.py --text "a hot air balloon made of glass" --workspace $DATA_DIR/exp2_balloon_2 -O --guidance controlnet --guidance_image_path scribbles/balloon.png --known_view_interval 5
python main.py --workspace $DATA_DIR/exp2_balloon_2 -O --test

python main.py --text "Bird with blue feathers and long wings" --workspace $DATA_DIR/exp2_bird_1 -O --guidance controlnet --guidance_image_path scribbles/bird.png --known_view_interval 5
python main.py --workspace $DATA_DIR/exp2_bird_1 -O --test

python main.py --text "Bird with rainbow feathers and long wings" --workspace $DATA_DIR/exp2_bird_2 -O --guidance controlnet --guidance_image_path scribbles/bird.png --known_view_interval 5
python main.py --workspace $DATA_DIR/exp2_bird_2 -O --test