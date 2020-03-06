import os
import numpy as np
import tensorflow as tf


class my_preWork():
    def __init__(self,test_size = 0.2,batch_size = 100):
        self.processed_folder = 'processed_face'
        self.test_size = test_size
        self.batch_size = batch_size


    def load_shuffle(self):
        self.image_list = []
        self.label_list = []
        pic_list = os.listdir(self.processed_folder)

        for i in range(len(pic_list)):
            path = os.path.join(self.processed_folder,pic_list[i])

            #f == female
            #m == male
            label = pic_list[i][1]
            if label == 'F':
                self.label_list.append(0)
                self.image_list.append(path)

            else:
                self.label_list.append(1)
                self.image_list.append(path)
        '''
        print(self.image_list)
        print()
        print(self.label_list)
        print(len(self.image_list))
        print(len(self.label_list))
        
        '''
        #form two-dimensional array
        temp = np.array([self.image_list,self.label_list])
        temp = temp.transpose()
        np.random.shuffle(temp)

        #print(temp)
        #train_test_split
        self.data = temp

        test_size = int(self.test_size * len(self.data))
        #print(test_size)
        self.test_set = self.data[0:test_size]
        self.train_set = self.data[test_size:]

        '''
        train = (self.train_set[0:4,0])

        print(train)
        label = self.train_set[0:4,1]
        print(label)
        '''
        return self.train_set,self.test_set


    def get_batch(self):
        train_set,_ = self.load_shuffle()
        batch_size = self.batch_size
        rand_index = np.random.choice(len(train_set), batch_size)
        #print(rand_index)
        batch = train_set[rand_index]
        # print(batch)
        return batch

    def get_test_batch(self):
        _, test_batch = self.load_shuffle()
        batch_size = self.batch_size
        rand_index = np.random.choice(len(test_batch), batch_size)
        # print(rand_index)
        test_batch = test_batch[rand_index]
        # print(batch)
        return test_batch

    def convert(self,pic_name):

        img_raw = tf.gfile.FastGFile(pic_name, 'rb').read()
        img = tf.image.decode_jpeg(img_raw)

        return img

    def splice_pic(self,choice = 0):

        if choice == 0:
            batch = self.get_batch()
        else:
            batch = self.get_test_batch()
        img1 = batch[0,0]
        img2 = batch[1,0]

        img1 = self.convert(img1)
        img2 = self.convert(img2)
        img = tf.concat([img1, img2], axis=0)
        times = int(self.batch_size-2)
        if times != 0:
            for i in range(times):
                img_x_name = batch[i+1,0]
                img_x = self.convert(img_x_name)
                img = tf.concat([img, img_x], axis=0)

        # print(img.eval(session=tf.Session()).shape)
        # print(self.batch[:,1])
        self.img = img
        self.label = batch[:,1]
        self.label = np.reshape(self.label,(self.batch_size,1))
        batch_list = np.reshape(np.array([x for x in range(self.batch_size)]),newshape=(self.batch_size,1))

        self.label = np.hstack([batch_list,self.label])

        #print(self.label)
        return self.img,self.label






if __name__ == '__main__':
        preWork = my_preWork(test_size=0.2,batch_size=100)

        preWork.splice_pic()

