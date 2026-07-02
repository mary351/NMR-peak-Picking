import os 


def replace_dataset_ext(path_to_imgs):
    
    for paths in path_to_imgs:

        folder_name = os.path.basename(os.path.dirname(paths))
        seed = folder_name.split("_")[-1]
        seed_tag = f"seed_{seed}"

        print(parent_folder)
        print(seed_tag)

        for filename in os.listdir(path):
            old_file = os.path.join(path, filename)

            if not os.path.isfile(old_file):
                continue

            name, ext = os.path.splitext(filename)
            new_name = f"{name}_{seed_tag}{ext}"

            os.rename(old_file, os.path.join(path, new_name))

            

    print("Added the corresponding seed name to each images in the folder")
