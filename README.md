This repository is for the Deep Learning (DL) Course for the Master's Course 'Data Science for Entrepreneurship and Business Innovation" at JADS. 
Specifically, with this project we are aiming to make a contribution to the current state of Machine Learning based visual recognition of birds (hence our cool name).
One especially difficult challenge in this field is to detecting sitting birds.

Our project aims to create an accurate and fast solution for the challenge at hand. 



**PROJECT REQUIREMENTS**

Running the code requires the following components:
- A Python environment with the packages listed in the requirements.txt file.
- A image-based dataset that is pre-split into train, test, and validation folders.
- For DETECTRON2 a Linux OS (and CUDA support) is required; For YOLO, any OS is fine (Tested on Windows and MacOS) and any processor can be used.
- The bird-detection-farm dataset is available on the repository. The scarecrow dataset is only available on request (is is part of the deliverable for the course).

Our code supports both COCO and YOLO data formats, next to our own custom format (of the original dataset).



**RUNNING THE CODE**

Please read the entire README file to get a complete understanding of the run requirements, known problems and how to solve them.
1. Navigate the the project folder (if you haven't already done so).
2. Create a virtual environment (e.g. python -m venv venv) and activate it (e.g. source venv/bin/activate (MacOS) or venv/Scipts/activate (Windows)).
3. Install the required packages by running pip install -r requirements.txt.
4. Run the code in the main.ipynb file. It uses the code in the utils.py files.
5. Have fun spotting birds!

NOTE: main.ipynb ONLY contains working code for YOLO. If you want to run the code for DETECTRON2, please run the code in the detectron2 folder - Including the required imports.



**NOTE FOR THE TEACHER**

- We have included the split datasets in the repository instead of the original dataset. Please refer to the split() function in the utils_support.py file for the function that splits the dataset.
- We have also included code for data augmentation and data loading with pytorch in the data_preparation.ipynb file. Because the models implemented those methods internally, we dO not use them in the main.ipynb file. They are, however, included for reference and future use.



**POTENTIAL PROBLEMS (AND SOLUTIONS)**

- In case you run into image corruption issues, please delete your labels.cache file in the data subfolders and rerun training.
- In case the dataset configuration of the YOLO model is not working, please check the .yaml file and make sure the paths are correct. It requires a full path to the data folder. 



**MEET THE ANGRY BIRDS TEAM**

Members:
- Nienke Reijnen: 2117034
- Andrea Ciavatti: 2115635
- Niels Boonstra: 1451294
- Yannick Lankhorst: 2052754
- Thom Zoomer: 2059225
- Anne Barnasconi: 2053988
