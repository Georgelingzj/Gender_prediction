import cv2
import os
import numpy as np



class Resize_detect(object):
    def __init__(self,pic=None):
        self.pic_name = pic
        self.save_path = 'processed_face'
    def print_pic_shape(self):

        img = cv2.imread(self.pic_name,0)
        #rint(img.shape)
        img_copy = img.copy()
        img_copy = cv2.resize(img_copy,(256,256),interpolation=cv2.INTER_AREA)
        self.image = img_copy
        #cv2.imwrite("human3_resize.jpg", img_copy)

    #based on opencv3 book
    def detect_face(self):
        img = self.image
        cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
        haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(haar_model)

        face = face_cascade.detectMultiScale(img,1.1,2)


        for (x,y,w,h) in face:
            img_after = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            '''
            print("x = {}".format(x))
            print("y = {}".format(y))
            print("w = {}".format(w))
            print("h = {}".format(h))
            print("another vertix")
            print("x+w = {}".format(x+w))
            print("y+h = {}".format(y+h))
            '''
            img_after = img_after[y:y+h, x:x+w]
        #img_after = img_after[x-w:x+w,y:y+2*h]
            img_after = cv2.resize(img_after, (28, 28), interpolation=cv2.INTER_AREA)
            #print(img_after.shape)

            _, pic_name_ = self.pic_name.rsplit('/', 1)
            # print(pic_name)
            # _,pic_name = (self.img_name).split('/')
            name,extension = pic_name_.split('.')
            pic_name_ = name + '_resized' + '.' + extension
            path = os.path.join(self.save_path, pic_name_)
            print(path)
            cv2.imwrite(path,img_after)


if __name__ == '__main__':
    work = Resize_detect(pic="original_face/human1.jpg")
    work.print_pic_shape()
    work.detect_face()