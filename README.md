Hardware Set-up:
     - Set camera opposite to Mirobot
     - Manually measure displacement vector, from the middle of the Mirobot base to the top left-hand corner of the camera frame using tape measure
     - Overwrite displacement values in main.py, must be in millimetres
     - Measure length of camera frame, again in millimetres, and replace value in mm_per_pixel found in bbox_to_mm function within camera.py
     - For moveObjectToBin() function within mirobot.py, set bin coordinates manually based on where bin is relative to arm
    
Software Requirements:
    - CUDA Runtime (11.0+), found here https://developer.nvidia.com/cuda-downloads 
    - ZED SDK, found here https://www.stereolabs.com/developers/release/
        - follow instructions for downloading ZED API within SDK folder downloaded 
    - CH340 Driver, found here https://www.wlkata.com/support/download-center 
    - Run pip install -r requirements.txt
