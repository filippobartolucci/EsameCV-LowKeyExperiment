import os 

dir_root = "./imgs"

if not os.path.exists(dir_root):
    raise Exception('Directory does not exist')

for img_name in os.listdir(dir_root):
    if img_name.endswith('_attacked.png') or img_name.endswith('_small.png'):
        os.remove(os.path.join(dir_root, img_name))