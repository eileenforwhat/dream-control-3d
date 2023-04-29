# Run scripts for exp3 (--guidance controlnet-zero123) results
DATA_DIR=/home/eileen/code/16825_3d_s23/dream-control-3d/output
DEVICE=0

python main.py --text "a wooden chair with cushions on the seat" --workspace $DATA_DIR/exp3_chair -O --guidance controlnet-zero123 --guidance_image_path scribbles/chair.png --device $DEVICE
python main.py --workspace $DATA_DIR/exp3_chair -O --test

python main.py --text "a wooden chair with cushions on the seat" --workspace $DATA_DIR/exp3_chair_2 -O --guidance controlnet-zero123 --guidance_image_path scribbles/chair_2.png --device $DEVICE
python main.py --workspace $DATA_DIR/exp3_chair_2 -O --test

python main.py --text "a wooden chair with cushions on the seat" --workspace $DATA_DIR/exp3_couch -O --guidance controlnet-zero123 --guidance_image_path scribbles/couch.png --device $DEVICE
python main.py --workspace $DATA_DIR/exp3_couch -O --test

python main.py --text "a black horse" --workspace $DATA_DIR/exp3_horse -O --guidance controlnet-zero123 --guidance_image_path scribbles/horse.png --device $DEVICE
python main.py --workspace $DATA_DIR/exp3_horse -O --test

python main.py --text "a black and white zebra" --workspace $DATA_DIR/exp3_zebra -O --guidanace controlnet-zero123 --guidance_image_path scribbles/horse.png --device $DEVICE
python main.py --workspace $DATA_DIR/exp3_zebra -O --test

python main.py --text "a hot air balloon made of wood" --workspace $DATA_DIR/exp3_balloon_1 -O --guidance controlnet-zero123 --guidance_image_path scribbles/balloon.png --device $DEVICE
python main.py --workspace $DATA_DIR/exp3_balloon_1 -O --test

python main.py --text "a hot air balloon made of glass" --workspace $DATA_DIR/exp3_balloon_2 -O --guidance controlnet-zero123 --guidance_image_path scribbles/balloon.png --device $DEVICE
python main.py --workspace $DATA_DIR/exp3_balloon_2 -O --test

python main.py --text "Bird with blue feathers and long wings" --workspace $DATA_DIR/exp3_bird_1 -O --guidance controlnet-zero123 --guidance_image_path scribbles/bird.png --device $DEVICE
python main.py --workspace $DATA_DIR/exp3_bird_1 -O --test

python main.py --text "Bird with rainbow feathers and long wings" --workspace $DATA_DIR/exp3_bird_2 -O --guidance controlnet-zero123 --guidance_image_path scribbles/bird.png --device $DEVICE
python main.py --workspace $DATA_DIR/exp3_bird_2 -O --test
