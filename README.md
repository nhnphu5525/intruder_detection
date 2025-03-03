# PROJECT OVERVIEW  
Intruder Detection by 3D Barrier is an AI-powered system designed to detect unauthorized access in a predefined 3D intrusion zone within a shopping mall. The project uses a customed YOLOv11 model for human detection and DeepSORT for tracking individuals. Users can define custom intrusion zones dynamically by selecting points on the screen, and the system will construct a 3D region. The detection algorithm then determines if a person has entered an off-limits area based on predefined closing times. This solution enhances security by ensuring that intrusions are only flagged after business hours for each specific zone.  
# SETUP GUIDE  
## I. Prerequisites  
Ensure that your system meets the following requirements:  
•	Python 3.8+  
•	CUDA-compatible GPU (Optional for better performance)  
•	Required dependencies: (Phus để file requirements ở đou thì viết vào nka Phus)  
## II. Installation Steps  
### 1. Clone the Repository  
### 2. Install Dependencies  
### 3. Ensure PyTorch Supports CUDA (Optional for GPU Acceleration)  
### 4. Download the HumanDetection Model  
### 5. Run the Application  
# User Guide  
## 1. Selecting an Intrusion Zone  
•	When the application starts, a window will appear displaying the video feed.  
•	Left-click to select points defining the bottom of the intrusion zone.  
•	Right-click to remove the last selected point if needed.  
•	Press C to confirm the zone and create a corresponding 3D intrusion region.   
## 2. Assigning Operating Hours  
## 3. Detecting Intrusions  
## 4. Exiting the Application





