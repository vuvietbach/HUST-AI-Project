# install dependencies
# first please install all dependencies by your self
# if you make change to FER please rerun pip install -e .  (e mean edit)
pip install -e .
# train script
python tools/train.py
# test script
python tools/test.py

python tools/demo.py


## IMPORTANT ############################
# read a video then output a precessed video
python tools/demo_video.py --input demo_images/movie.mp4 --output demo_images/out.avi
# read a from webcam 
python tools/demo_video.py --input webcam
