import cv2
import sys
import easyocr
import os

img_path_f = sys.argv[1]
folder_path = img_path_f

# Get a list of all files in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Iterate over each image file
for img_filename in image_files:
    img_path = os.path.join(folder_path, img_filename)

    print("Processing image:", img_path)

    # Run OCR on the current image
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img_path)

    # Print OCR result
    print(result)



