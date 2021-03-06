# Standard
import glob
import shutil
import os


# Directories
SRC_DIR = "data/geological_similarity"
DST_DIR = "static/images"

 

def rename_images():
    """ renames images in source directory: the new name contains
     both the class and the image id
    """
    # loop through the directory to rename all the files
    for classe in os.listdir(SRC_DIR):
        SRC_IMG = os.path.join(SRC_DIR, classe)
        for imgname in os.listdir(SRC_IMG):
            new =classe + imgname  # new file name
            src = os.path.join(SRC_IMG, imgname)  # file source
            dst = os.path.join(SRC_IMG, new)  # file destination
            # rename all the file
            os.rename(src, dst)
    return "renaming done with sucess!"


def to_static_folder():
    """ Sends images from source directory to a target directory"""
    for classe in os.listdir(SRC_DIR):
        SRC_IMG = os.path.join(SRC_DIR, classe)
        for f in glob.iglob(os.path.join(SRC_IMG, "*.jpg")):
            shutil.copy(f, DST_DIR)
    return "images copied with sucess"



if __name__ == "__main__":
    rename_images()
    to_static_folder()