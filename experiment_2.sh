python main.py --text "a wooden chair with cushions on the seat" --workspace trial -O --guidance_image_path scribbles/chair.png
python main.py --workspace /mnt/data/exp1_chair -O --test

python main.py --text "a wooden chair with cushions on the seat" --workspace trial -O --guidance_image_path scribbles/chair_2.png
python main.py --workspace /mnt/data/exp1_chair_2 -O --test

python main.py --text "a wooden chair with cushions on the seat" --workspace trial -O --guidance_image_path scribbles/couch.png
python main.py --workspace /mnt/data/exp1_couch -O --test