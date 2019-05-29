import cv2
import time
import os
import numpy as np
from sklearn.cluster import KMeans

class detection:    
    def __init__(self):
        self.camera_width = 640
        self.camera_height = 480
        self.img_ratio = 80/640

        self.fps = ""
        self.vidfps = 15
        self.elapsedTime = 0
        self.message = "Push [p] to take a background picture."
        self.flag_detection = False

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, self.vidfps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        #self.fourcc = cv2.VideoWriter_fourcc(*'XVID')##movie save
        #self.writer = cv2.VideoWriter('output.avi',self.fourcc, self.vidfps,
        #                              (self.camera_width,self.camera_height))##movie save

        time.sleep(1)

    def resize(self, img):
        img = cv2.resize(img, (int(self.img_ratio*self.camera_width), int(self.img_ratio*self.camera_height)))
        return img

    def MOG_map(self, img1, img2):
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)
        img2 = cv2.GaussianBlur(img2, (5, 5), 0)
        fgmask = fgbg.apply(np.uint8(img1))
        fgmask = fgbg.apply(np.uint8(img2))
        return fgmask

    def k_means(self, map):
        Y, X = np.where(map > 200)
        if len(Y) > 1:
            result = KMeans(n_clusters=2, random_state=0).fit_predict(np.array([X,Y]).T)
        else:
            result = []
        return Y, X, result 

    def get_x_y_limit(self, Y, X, result, cluster):
        NO = np.where(result==cluster)
        x_max = np.max(X[NO])
        x_min = np.min(X[NO])
        y_max = np.max(Y[NO])
        y_min = np.min(Y[NO])

        x_max = int(x_max/self.img_ratio)
        x_min = int(x_min/self.img_ratio)
        y_max = int(y_max/self.img_ratio)
        y_min = int(y_min/self.img_ratio)
        return x_min, y_min, x_max, y_max

    def bounding_box(self, img, x_min, y_min, x_max, y_max):
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
        return img

    def main(self):
        while self.cap.isOpened():
            t1 = time.time()

            ret, image = self.cap.read()
            if not ret:
                break

            key = cv2.waitKey(1)&0xFF

            # Start object detection.
            if key == ord('p'):
                background_img = self.resize(image)
                self.message = "Start object detection."
                self.flag_detection = True

            # quit
            if key == ord('q'):
                #self.writer.release()##movie save
                break
            
            # Start object detection.
            if self.flag_detection == True:
                img = self.resize(image)
                img = self.MOG_map(img, background_img)
                
                if np.max(img) >200:
                    Y, X, result = self.k_means(img)

                    if len(result) > 1:
                        x_min, y_min, x_max, y_max = self.get_x_y_limit(Y, X, result, 0)
                        image = self.bounding_box(image, x_min, y_min, x_max, y_max)
                        x_min, y_min, x_max, y_max = self.get_x_y_limit(Y, X, result, 1)
                        image = self.bounding_box(image, x_min, y_min, x_max, y_max)

            # message
            cv2.putText(image, self.message, (self.camera_width - 200, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, self.fps, (self.camera_width - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 255, 0 ,0), 1, cv2.LINE_AA)

            cv2.imshow("Result", image)
            #self.writer.write(image)##movie save

            # FPS
            elapsedTime = time.time() - t1
            self.fps = "{:.0f} FPS".format(1/elapsedTime)


        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main = detection()
    main.main()