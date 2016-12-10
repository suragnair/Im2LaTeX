from PIL import Image
from sys import argv
import os

MAX_WIDTH = 280
MAX_HEIGHT = 50

def pad_resize(source_folder_path, dest_folder_path):
	# take all pngs from source_folder_path and if height and width are less than max values
	# pad smaller ones to max size and add them to dest_folder_path
	# save a list of ignored files

	files = [x for x in os.listdir(source_folder_path) if x.endswith('png')]
	ignored = open(dest_folder_path + 'ignored.txt', 'w')

	for file in files :
		img = Image.open(source_folder_path + file)

		if img.size[0] <= MAX_WIDTH and img.size[1] <= MAX_HEIGHT:
			new_im = Image.new("RGB", (MAX_WIDTH, MAX_HEIGHT), "white")
			new_im.paste(img, ((MAX_WIDTH -img.size[0])/2,(MAX_HEIGHT-img.size[1])/2))
			new_im.save(dest_folder_path + file)

		else:
			ignored.write(file + '\n')

		img.close()
	
	ignored.close()

if __name__ == "__main__":
	_, source_folder_path, dest_folder_path = argv
	pad_resize(source_folder_path, dest_folder_path)
