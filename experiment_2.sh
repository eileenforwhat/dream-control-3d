# python main.py --text "a wooden chair with cushions on the seat" --workspace /mnt/data/exp1_chair -O --guidance_image_path scribbles/chair.png
# python main.py --workspace /mnt/data/exp1_chair -O --test

python main.py --text "a wooden chair with cushions on the seat" --workspace /mnt/data/exp1_chair_2 -O --guidance_image_path scribbles/chair_2.png
python main.py --workspace /mnt/data/exp1_chair_2 -O --test

python main.py --text "a wooden chair with cushions on the seat" --workspace /mnt/data/exp1_couch -O --guidance_image_path scribbles/couch.png
python main.py --workspace /mnt/data/exp1_couch -O --test

python main.py --text "a black horse" --workspace /mnt/data/exp1_horse -O --guidance_image_path scribbles/horse.png
python main.py --workspace /mnt/data/exp1_horse -O --test

python main.py --text "a black and white zebra" --workspace /mnt/data/exp1_zebra -O --guidance_image_path scribbles/horse.png
python main.py --workspace /mnt/data/exp1_zebra -O --test

python main.py --text "a hot air balloon made of wood" --workspace /mnt/data/exp1_balloon_1 -O --guidance_image_path scribbles/balloon.png
python main.py --workspace /mnt/data/exp1_balloon_1 -O --test

python main.py --text "a hot air balloon made of glass" --workspace /mnt/data/exp1_balloon_2 -O --guidance_image_path scribbles/balloon.png
python main.py --workspace /mnt/data/exp1_balloon_2 -O --test

python main.py --text "Bird with blue feathers and long wings" --workspace /mnt/data/exp1_bird_1 -O --guidance_image_path scribbles/bird.png
python main.py --workspace /mnt/data/exp1_bird_1 -O --test

python main.py --text "Bird with rainbow feathers and long wings" --workspace /mnt/data/exp1_bird_2 -O --guidance_image_path scribbles/bird.png
python main.py --workspace /mnt/data/exp1_bird_2 -O --test