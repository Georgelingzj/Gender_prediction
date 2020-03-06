import os
from aligned_1 import align
from resize import Resize_detect

print("reading photos---------")

#define path
original_face_path = 'original_face'
#Include images with qualified extensions
processed_face_path = 'processed_face'

original_face_name_list = os.listdir(original_face_path)
total = len(original_face_name_list)
print(total)
base_dir = os.path.dirname(os.path.abspath(original_face_path))
base_dir = os.path.join(base_dir,original_face_path)
disqualified_num = 0
for i in range(total):
    _,extension = original_face_name_list[i].split('.')
    #print(name,extension)
    if extension != 'jpg':
        print("No.{}, failed".format(i+1))
        path =  os.path.join(base_dir,original_face_name_list[i])
        os.remove(path)
    # else:
    #     #print("No.{}, succeeded".format(i+1))

print()
print("There are total {} photos and there are {} photos are disqualified".format(total,disqualified_num))

# delete old pics in processed folder


original_face_name_list = os.listdir(original_face_path)
for j in range(len(original_face_name_list)):
    path = os.path.join(base_dir, original_face_name_list[j])
    #print(path)
    alinged = align(path)
    alinged.load_pic()
    is_valid = alinged.detect_face()
    if is_valid == 0:
        continue
    else:
        alinged.rotate_whole()


print()
print("aligned faces finished --------")
print()
print("begin to resize pics---------")
print()

processed_face_name_list = os.listdir(processed_face_path)
total_2 = len(processed_face_name_list)
base_dir_2 = os.path.dirname(os.path.abspath(processed_face_path))
base_dir_2 = os.path.join(base_dir_2,processed_face_path)

#if emptry
if total_2 == 0:
    print("empty folder")
else:
    for k in range(total_2):
        print("Woring on N0.{} photo".format(k+1))
        path = os.path.join(base_dir_2,processed_face_name_list[k])
        resizer = Resize_detect(path)
        resizer.print_pic_shape()
        resizer.detect_face()
        os.remove(path)




print("Resize finish")
print()
print("begin to train!-------")


