# HandTalk-Real-time-Recognition-of-American-Sign-Language
This is a repository for of the projects in Machine Vision. The system makes use of the Mediapipe and OpenCV libraries.

Steps to run the project:


1. Clone the repository.\
 `git clone https://github.com/shreyadm/HandTalk-Real-time-Recognition-of-American-Sign-Language.git`
2. Move to the project folder.\
   `cd HandTalk-Real-time-Recognition-of-American-Sign-Language`
3. Install the requirements.\
    `pip install -r requirements.txt`
4. Collect desired dataset images.\
   `python collect_imgs.py`
6. Create dataset.\
   `python create_dataset.py`
8. Train the model.\
   `python train_classifier.py`
9. Run the real-time Sign Language Recognition Project.\
  `python inference_classifier.py`
