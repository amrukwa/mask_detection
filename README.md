# Mask detection model
![mask_detection](static/mas_example.jpg)  
This project aims to create a simple mask detection model with the usage of openCV and scikit learn.
## The scheme
Before finding masks in the picture, the program finds the faces first by the openCV Cascade Classifier. Then, cropped images are forwarded into the Random Forest Classifier. After being labeled as masked, unmasked or incorrectly masked, pictures are shown with the marked region and text annotation. All of the steps can be seen in the `workflow.ipynb` file.
## The data
As I wanted the model to be able to detect the third class - incorrect mask, I used [this](https://www.kaggle.com/spandanpatnaik09/face-mask-detectormask-not-mask-incorrect-mask) dataset. For showing the results on the different image I used the picture available [here](https://www.kaggle.com/andrewmvd/face-mask-detection).
## To be done
I plan on deploying the model (with preprocessing as well) using Flask and Docker.
## Concerns
As the whole mask detection relies on the prior face detection, the results may be inaccurate when most of the face is obscurred (not only mask, but also glasses or huge hat on the head). Moreover, I noticed the model may also fail when the person's face is mostly facing to the side or is blurred. All of this is the result of the pretrained model I used.
## What else could be done?
The whole project could be extended for real time detection on the video using openCV. This would help to control the compliance with the sanitary regulations.