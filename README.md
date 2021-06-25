## About the Project
This project aims to perform face detection on the members of TWICE, a famous K-pop girl group, using Python and OpenCV library.

Face detection itself is a technique that identifies or locates human faces in images. For example, when we want to take photos using our smartphones, they will instantly detects faces in the photos. It is imporant to note that Face Detection is very much different from Face recognition. Face Detection merely acknowledges the presence of faces in images while Face Recognition involves identifying the identity of the faces.

Face detection is performed using a classifier which is an algorithm that decides whether a given image is a positive image (with faces) or a negative one (without a face). Therefore, a classifier needs to be trained with countless of images so that it can perform well. Fortunately, OpenCV comes with a lot of pre-trained classsifiers such as classifiers for eyes, faces, smiles, etc. These classifiers come in the form of xml files and are located in opencv/data/haarcascades/ folder. 

## Notes
The pre-trained classifier that we will use for this project and the pictures of members of TWICE are included. 
