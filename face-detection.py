import os
from os import listdir
import cv2
import itertools
from PIL import Image
import numpy as np

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of each person
    '''
    path_directories = [f for f in listdir(root_path)]
    return path_directories

def get_class_id(root_path, train_names):
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''

    image_list = []
    image_classes_list = []

    for train_class, train_name in enumerate(train_names):
        folder_path = os.path.join(root_path, train_name)

        for image_path in os.listdir(folder_path):
            image = cv2.imread(os.path.join(folder_path, image_path))

            if image is not None:
                image_list.append(image)
                image_classes_list.append(train_class)

    return image_list, image_classes_list

def detect_train_faces_and_filter(image_list, image_classes_list):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one
        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered image classes id
    '''
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    train_face_gray_list = [] 
    train_face_classes_list = []

    for img, img_class in zip(image_list, image_classes_list):
        if img.dtype != "uint8":
            img = (img * 255).round().astype(np.uint8)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
        
        if(len(detected_faces) < 1):
            continue
        
        for face_rect in detected_faces:
            x, y, w, h = face_rect
            face_img = img_gray[y:y+w, x:x+h]
            train_face_gray_list.append(face_img)
            train_face_classes_list.append(img_class)
        
    return train_face_gray_list, train_face_classes_list

def train(train_face_grays, image_classes_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(train_face_grays, np.array(image_classes_list))

    return face_recognizer

def get_test_images_data(test_root_path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all image in the test directories
    '''

    image_list = []
    
    for image_path in os.listdir(test_root_path):
        image = cv2.imread(os.path.join(test_root_path, image_path))

        if image is not None:
            image_list.append(image)

    return image_list

def detect_test_faces_and_filter(image_list):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
    '''
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    test_face_gray_list = [] 
    test_face_rects_list = []

    for index, img in enumerate(image_list):
        if img.dtype != "uint8":
            img = (img * 255).round().astype(np.uint8)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
        
        if(len(detected_faces) < 1):
            continue
        
        for face_rect in detected_faces:
            x, y, w, h = face_rect
            face_img = img_gray[y:y+w, x:x+h]
            test_face_gray_list.append(face_img)
            test_face_rects_list.append(face_rect)
    
    return test_face_gray_list, test_face_rects_list

def predict(recognizer, test_faces_gray):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        test_faces_gray : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''

    result_list = []
    for image in test_faces_gray:
        result, confidence = recognizer.predict(image)
        result_list.append(result)
    
    return result_list

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, size):
    '''
        To draw prediction results on the given test images and resize the image

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories
        size : number
            Final size of each test image

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''
    drawn_result = []

    for index, image in enumerate(test_image_list):
        x, y, w, h = test_faces_rects[index]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 3)
        res = predict_results[index]
        text = train_names[res]
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
        # cv2.imshow("",image)
        # cv2.waitKey(0)
        drawn_result.append(image)
    return drawn_result

def combine_and_show_result(image_list, size):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
        size : number
            Final size of each test image
    '''
    combined_image = Image.new('RGB', (size * len(image_list), size))
    xOffset = 0
    for image in image_list:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        converted_img = Image.fromarray(np.uint8(image)).convert('RGB')
        combined_image.paste(converted_img, (xOffset,0))
        xOffset += size
    
    combined_image.show()

if __name__ == "__main__":

    '''
    Please modify train_root_path value according to the location of
    your data train root directory

    -------------------
    Modifiable
    -------------------
    '''
    train_root_path = os.path.join(os.path.join(os.path.abspath(os.getcwd()),'dataset'),'train')
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, filtered_classes_list = detect_train_faces_and_filter(train_image_list, image_classes_list)  
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = os.path.join(os.path.join(os.path.abspath(os.getcwd()), 'dataset'), 'test')
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects = detect_test_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, 200)
    combine_and_show_result(predicted_test_image_list, 200)