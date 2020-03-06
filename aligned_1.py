import cv2
import face_recognition
import math
import os

class align():
    def __init__(self,pic_name):
        self.img_name = pic_name
        self.save_path = 'processed_face'
    def load_pic(self):
        img_name = self.img_name
        #define pic
        self.img = cv2.imread(img_name,1)

    def detect_face(self):
        face_landmark = face_recognition.face_landmarks(self.img,model="large")

        if len(face_landmark) == 0:
            return 0
        face_landmark_dict = face_landmark[0]
        #print(face_landmark_dict)
        self.landmark = face_landmark_dict

        right_eye_list = face_landmark_dict['right_eye']
        left_eye_list = face_landmark_dict['left_eye']

        right_eye_centre_x = 0
        right_eye_centre_y = 0

        left_eye_centre_x = 0
        left_eye_centre_y = 0

        for i in range(len(right_eye_list)):
            right_eye_centre_x += right_eye_list[i][0]
            right_eye_centre_y += right_eye_list[i][1]

        for j in range(len(left_eye_list)):
            left_eye_centre_x += left_eye_list[i][0]
            left_eye_centre_y += left_eye_list[i][1]

        right_eye_centre_x,right_eye_centre_y = right_eye_centre_x/float(len(right_eye_list)),right_eye_centre_y/float(len(right_eye_list))
        left_eye_centre_x, left_eye_centre_y = left_eye_centre_x/float(len(left_eye_list)), left_eye_centre_y/float(len(left_eye_list))

        #print("right is {}".format((right_eye_centre_x,right_eye_centre_y)))
        #print("left is {}".format((left_eye_centre_x,left_eye_centre_y)))

        direction = 'clockwise'
        if right_eye_centre_y>left_eye_centre_y:
            #C_x,C_y = 0,right_eye_centre_y
            direction = 'counterclockwise'



        AB_length = math.sqrt(math.pow((right_eye_centre_x-left_eye_centre_x),2)+math.pow((right_eye_centre_y-left_eye_centre_y),2))
        AC_length = right_eye_centre_x
        BC_length = math.sqrt(math.pow(left_eye_centre_x,2)+math.pow((left_eye_centre_y-right_eye_centre_y),2))

        cosA = float((math.pow(AB_length,2)+math.pow(AC_length,2)-math.pow(BC_length,2)))/float(2*AB_length*AC_length)
        #print(cosA)
        #human head angle
        if cosA > 1.0 or cosA < -1.0:
            return 0

        angle = (math.acos(cosA))*(180/float(math.pi))
        print(angle,direction)

        self.angle = angle
        self.direction = direction
        return 1

    def rotate_whole(self):
        image = self.img

        rows, cols = image.shape[0:2]

        if self.direction == 'clockwise':
            M = cv2.getRotationMatrix2D((rows / 2,cols / 2),-self.angle,1)
        else:
            M = cv2.getRotationMatrix2D((rows / 2, cols / 2), self.angle, 1)

        img_after = cv2.warpAffine(image,M,(rows,cols))

        _,pic_name = self.img_name.rsplit('/',1)
        #print(pic_name)
        #_,pic_name = (self.img_name).split('/')
        path = os.path.join(self.save_path, pic_name)

        #print(path)
        cv2.imwrite(path, img_after)



#
# if __name__ == '__main__':
#     align_ = align('original_face/human4.jpg')
#     align_.load_pic()
#     align_.detect_face()
#     align_.rotate_whole()

