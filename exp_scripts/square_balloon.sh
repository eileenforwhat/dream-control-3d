DATA_DIR=/mnt/data

python main.py --text "a hot air balloon made of wood" --workspace $DATA_DIR/baseline_square_balloon -O
python main.py --workspace $DATA_DIR/baseline_square_balloon -O --test

python main.py --text "a hot air balloon made of wood" --workspace $DATA_DIR/exp1_balloon_square -O --guidance controlnet --guidance_image_path scribbles/square_balloon.png
python main.py --workspace $DATA_DIR/exp1_balloon_square -O --test

python main.py --text "a hot air balloon made of wood" --workspace $DATA_DIR/exp2_balloon_square -O --guidance controlnet --guidance_image_path scribbles/square_balloon.png  --guidance_image_view 0 --guidance_view_loss_factor 2.0
python main.py --workspace $DATA_DIR/exp2_balloon_square -O --test

python main.py --text "a hot air balloon made of wood" --workspace $DATA_DIR/exp2_2_balloon_square -O --guidance controlnet --guidance_image_path scribbles/square_balloon.png  --guidance_image_view 0 --guidance_view_loss_factor 2.0 --controlnet_conditioning_scale 1.0
python main.py --workspace $DATA_DIR/exp2_2_balloon_square -O --test

