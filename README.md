# Lip reading model

**Dataset used is taken from this link**
<a href = "https://sites.google.com/site/achrafbenhamadou/-datasets/miracl-vc1"> lip reading dataset</a>
<p>
  Here is the dataset description:</br></br>
  MIRACL-VC1 is a lip-reading dataset including both depth and color images. It can be used for diverse research fields like visual speech recognition, face detection, and          biometrics. Fifteen speakers (five men and ten women) positioned in the frustum of an MS Kinect sensor and utter ten times a set of ten words and ten phrases (see the table below). Each instance of the dataset consists of a synchronized sequence of color and depth images (both of 640x480 pixels).  The MIRACL-VC1 dataset contains a total number of 3000 instances.</br></br>
  The uttered words are:
  <ol>
  <li>Begin</li>
  <li>Choose</li>
  <li>Connection</li>
  <li>Navigation</li>
  <li>Next</li>
  <li>Previous</li>
  <li>Start</li>
  <li>Stop</li>
  <li>Hello</li>
  <li>Web</li>
  </ol>
  </br></br>
  The uttered phrases are:</br>
  <ol>
  <li>Stop navigation.</li>
  <li>Excuse me.</li>
  <li>I am sorry.</li>
  <li>Thank you.</li>
  <li>Good bye.</li>
  <li>I love this game.</li>
  <li>Nice to meet you.</li>
  <li>You are welcome.</li>
  <li>How are you?</li>
  <li>Have a good time.</li>
  </ol>
  
  
  Details and results for visual speech recognition can be found in the following papers 

Ahmed Rekik, Achraf Ben-Hamadou, Walid Mahdi: A New Visual Speech Recognition Approach for RGB-D Cameras. ICIAR 2014: 21-28 
<a href = "https://www.researchgate.net/profile/Achraf_Ben-Hamadou/publication/263409473_A_New_Visual_Speech_Recognition_Approach_For_RGB-D_Cameras/links/544422310cf2a76a3ccd6bcb?ev=pub_int_doc_dl&origin=publication_detail&inViewer=true">pdf</a></br>

Folder architecture of the dataset is explained as follows:</br>
![image](https://user-images.githubusercontent.com/47064840/114566772-3696ac80-9c90-11eb-8416-4a4ba267ece2.png)
</br></br>

Now to train our model we first need to process our data and only pass the necessary data to the training set.

A sample of dataset we have :

![image](https://user-images.githubusercontent.com/47064840/114567463-e5d38380-9c90-11eb-9fa4-c105ed8656c3.png)


So, as we can see the dataset contain the full body image of a person, but we are just interested in the lips part to record their movement and train our model accordingly.

To do this part we took help of <a href="https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/">this link</a>

The tools we are using to filter out lip image from a full body image are:
- <a href = "http://dlib.net/">dlib</a> Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems. It is used in both industry and academia in a wide range of domains including robotics, embedded devices, mobile phones, and large high performance computing environments.</br>
- <a href = "https://opencv.org/">openCV</a> OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in the commercial products. Being a BSD-licensed product, OpenCV makes it easy for businesses to utilize and modify the code.</br>
- <a href="https://www.python.org/">Python</a> Python is an interpreted, high-level and general-purpose programming language. </br>

**How to do this:** </br></br>
Facial landmark indexes for face regions
The facial landmark detector implemented inside dlib produces 68 (x, y)-coordinates that map to specific facial structures. These 68 point mappings were obtained by training a shape predictor on the labeled iBUG 300-W dataset.

Below we can visualize what each of these 68 coordinates map to:

![image](https://user-images.githubusercontent.com/47064840/114570754-b8d4a000-9c93-11eb-8cf4-1227106246a6.png)

</br></br>

Examining the image, we can see that facial regions can be accessed via simple Python indexing (assuming zero-indexing with Python since the image above is one-indexed):

- The mouth can be accessed through points [48, 68].
- The right eyebrow through points [17, 22].
- The left eyebrow through points [22, 27].
- The right eye using [36, 42].
- The left eye with [42, 48].
- The nose using [27, 35].
- And the jaw via [0, 17].

These mappings are encoded inside the FACIAL_LANDMARKS_IDXS dictionary inside <a href="https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils.py#L8">face_utils</a> of the imutils library:

**Visualizing facial landmarks with OpenCV and Python**\

A slightly harder task is to visualize each of these facial landmarks and overlay the results on an input image.

To accomplish this, we’ll need the visualize_facial_landmarks function, <a href="https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/helpers.py#L56">already included in the imutils library</a>:

Our visualize_facial_landmarks function requires two arguments, followed by two optional ones, each detailed below:

- image : The image that we are going to draw our facial landmark visualizations on.
- shape : The NumPy array that contains the 68 facial landmark coordinates that map to various facial parts.
- colors : A list of BGR tuples used to color-code each of the facial landmark regions.
- alpha : A parameter used to control the opacity of the overlay on the original image.

![image](https://user-images.githubusercontent.com/47064840/114576405-bb85c400-9c98-11eb-924d-ca0f63c4ea66.png)
</br>
</br>
![image](https://user-images.githubusercontent.com/47064840/114576441-c5a7c280-9c98-11eb-8d6a-2396b0c57679.png)

**Extracting parts of the face using dlib, OpenCV, and Python**

Before you continue, make sure you have:

Installed dlib according to my instructions in this blog post.
Have installed/upgraded imutils to the latest version, ensuring you have access to the face_utils submodule: pip install --upgrade imutils


