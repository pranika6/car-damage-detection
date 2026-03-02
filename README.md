# Car Damage Detection 
### ML-Based Image Classification & Computer Vision System

An automated deep learning platform designed to detect, localize, and classify vehicle damage from images. This system streamlines the vehicle inspection process, significantly reducing the time required for insurance claim assessments.

---

##  Project Overview (Detailed)
The traditional process of assessing vehicle damage is manual, time-consuming, and prone to human error. This project leverages **Computer Vision** to automate the initial inspection phase.

**The Workflow:**
* **Image Acquisition:** The user uploads a photo of the damaged vehicle.
* **Pre-processing:** The system uses **OpenCV** to resize, normalize, and highlight structural contours of the car.
* **Feature Extraction:** A Convolutional Neural Network (CNN) identifies patterns associated with dents, scratches, and shattered glass.
* **Classification:** The model categorizes the damage type (e.g., bumper dent, door scratch, broken headlamp) and provides a confidence score.

**Impact:** By integrating this model into insurance workflows, companies can automate "green-channel" claims—instantly approving minor repairs and flagging major damages for human review.

---

##  Features
* **Automated Classification:** Identifies multiple damage types including dents, scratches, and glass breakage.
* **Image Pre-processing Pipeline:** Uses OpenCV for noise reduction and edge detection to improve model accuracy.
* **High-Precision Inference:** Built on a Deep Learning architecture (CNN) optimized for image-based feature extraction.
* **Visual Confidence Mapping:** Generates Matplotlib visualizations to show the model's prediction confidence for each upload.
* **Scalable Architecture:** Designed to be integrated as a backend API for mobile insurance applications.

---

##  Technologies Used
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Data Manipulation:** NumPy
* **Visualization:** Matplotlib
* **Version Control:** Git

---

##  Data Handling & Model Logic
* **Data Augmentation:** Implemented rotation, flipping, and zooming to increase model robustness against varying lighting and angles.
* **CNN Architecture:** Utilized multiple convolutional layers followed by Max-Pooling and Dropout layers to prevent overfitting.
* **Optimization:** Used the Adam optimizer and Categorical Cross-Entropy loss function for multi-class classification.



---
