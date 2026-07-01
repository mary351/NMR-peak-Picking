import os 


def replace_dataset_ext(path_to_imgs):
    
    for paths in path_to_imgs:

        folder_name = os.path.basename(os.path.normpath(paths))
        seed = folder_name.split("_")[2]
        seed_tag = f"seed_{seed}"

        img_names = os.listdir(paths)
        for fil in img_names:
            name, ext = os.path.splittext(fil)
            new_name = f"{name}_{seed_tag}{ext}"
            os.rename(os.path.join(paths, fil), os.path.join(paths, new_name))

    print("Added the corresponding seed name to each images in the folder")