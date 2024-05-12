from os import listdir, mkdir
from os.path import isfile, join, dirname, exists
from PIL import Image
from preprocessingutils import preprocess,extractImageLines
import os

def process_image(image_path, output_dir):
    # print("fonts-dataset//IBM Plex Sans Arabic//0.jpeg")
    print("image processed currently",os.path.splitext(os.path.basename(image_path))[0])
    img = preprocess(image_path)  # Assuming preprocess returns a NumPy array
    # img = Image.fromarray(img)  # Convert to PIL Image object
    # new_path = join(output_dir, image_path.split("/")[-1])
    # img.save(new_path)
    extractImageLines(img,os.path.splitext(os.path.basename(image_path))[0],output_dir,image_path)



def process_folder(folder_path):
  # output_dir = join(dirname(folder_path), "processed")
  # if not exists(output_dir):
  #   mkdir(output_dir)  # Create output folder if it doesn't exist
  # for f in listdir(folder_path):
  #   print(f)
  print
  for folder in listdir(folder_path):
    img = join(folder_path, folder)
    print(img)
    if isfile(img) and img.lower().endswith(".jpeg"):
      # Check for image files with common extensions
      process_image(img, "Processed-fonts-dataset\Scheherazade New")

# Replace "path/to/your/folders" with the actual path containing your image folders
# for folder in listdir("fonts-dataset"):
  # folder_path = join("fonts_dataset", folder)
  # print(folder_path)
  # if not isfile(folder_path):  # Check if it's a folder
  #   process_folder(folder_path)
process_folder("fonts-dataset\Scheherazade New")

print("Image processing complete!")
