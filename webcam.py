import cv2
import numpy as np
from keras.models import load_model
from skimage.transform import resize
from PIL import Image

model = load_model('CNNmodel.h5')


def prediction(pred):
    return(chr(pred+ 65))


def keras_predict(model, image):
    data = np.asarray( image, dtype="int32" )
    
    pred_probab = model.predict(data)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

'''def keras_process_image(img):
    
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (1,28,28), interpolation = cv2.INTER_AREA)
  
    return img
''' 

def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

def main():
    l = []
    sentence = ""
    while True:
        
        cam_capture = cv2.VideoCapture(0)
        cam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        cam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
        _, image_frame = cam_capture.read()  

        # control button
        c = cv2.waitKey(1) & 0xFF

        # Clear the sentence
        if c == ord('c') or c == ord('C'):
            sentence = ""

        # Put Space between words
        if c == ord('s') or c == ord('S'):
            sentence += " "
        
        # Select ROI
        im2 = crop_image(image_frame, 300,300,300,300)
        image_grayscale = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
        image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (15,15), 0)
        im3 = cv2.resize(image_grayscale_blurred, (28,28), interpolation = cv2.INTER_AREA)

        im4 = np.resize(im3, (28, 28, 1))
        im5 = np.expand_dims(im4, axis=0)

        pred_probab, pred_class = keras_predict(model, im5)
    
        curr = prediction(pred_class)

        # Put Space between words
        if True:
            cv2.putText(image_frame, curr, (600, 250), cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
            if c == ord('a') or c == ord('A'):
                sentence += " "
        
        cv2.putText(image_frame, sentence, (600, 250), cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
 
        # Display cropped image
        cv2.rectangle(image_frame, (300, 300), (600, 600), (255, 255, 00), 3)
        cv2.imshow("frame",image_frame)
        
        
        #cv2.imshow("Image4",resized_img)
        cv2.imshow("Image",image_grayscale_blurred)

        # Exit webcam
        if c == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()

cam_capture.release()
cv2.destroyAllWindows()
