# Run scripts for exp3 (--guidance controlnet-zero123) results
DATA_DIR=/mnt/data

python main.py --text "a wooden chair with cushions on the seat" --workspace $DATA_DIR/exp3_chair -O --guidance controlnet-zero123 --guidance_image_path scribbles/chair.png
python main.py --workspace $DATA_DIR/exp3_chair -O --test

python main.py --text "a wooden chair with cushions on the seat" --workspace $DATA_DIR/exp3_chair_2 -O --guidance controlnet-zero123 --guidance_image_path scribbles/chair_2.png
python main.py --workspace $DATA_DIR/exp3_chair_2 -O --test

python main.py --text "a wooden chair with cushions on the seat" --workspace $DATA_DIR/exp3_couch -O --guidance controlnet-zero123 --guidance_image_path scribbles/couch.png
python main.py --workspace $DATA_DIR/exp3_couch -O --test

python main.py --text "a black horse" --workspace $DATA_DIR/exp3_horse -O --guidance controlnet-zero123 --guidance_image_path scribbles/horse.png
python main.py --workspace $DATA_DIR/exp3_horse -O --test

python main.py --text "a black and white zebra" --workspace $DATA_DIR/exp3_zebra -O --guidanace controlnet-zero123 --guidance_image_path scribbles/horse.png
python main.py --workspace $DATA_DIR/exp3_zebra -O --test

python main.py --text "a hot air balloon made of wood" --workspace $DATA_DIR/exp3_balloon_1 -O --guidance controlnet-zero123 --guidance_image_path scribbles/balloon.png
python main.py --workspace $DATA_DIR/exp3_balloon_1 -O --test

python main.py --text "a hot air balloon made of glass" --workspace $DATA_DIR/exp3_balloon_2 -O --guidance controlnet-zero123 --guidance_image_path scribbles/balloon.png
python main.py --workspace $DATA_DIR/exp3_balloon_2 -O --test

python main.py --text "Bird with blue feathers and long wings" --workspace $DATA_DIR/exp3_bird_1 -O --guidance controlnet-zero123 --guidance_image_path scribbles/bird.png
python main.py --workspace $DATA_DIR/exp3_bird_1 -O --test

python main.py --text "Bird with rainbow feathers and long wings" --workspace $DATA_DIR/exp3_bird_2 -O --guidance controlnet-zero123 --guidance_image_path scribbles/bird.png
python main.py --workspace $DATA_DIR/exp3_bird_2 -O --test
