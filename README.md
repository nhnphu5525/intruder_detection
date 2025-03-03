# PROJECT OVERVIEW  
Intruder Detection by 3D Barrier is an AI-powered system designed to detect unauthorized access in a predefined 3D intrusion zone within a shopping mall. The project uses a customed YOLOv11 model for human detection and DeepSORT for tracking individuals. Users can define custom intrusion zones dynamically by selecting points on the screen, and the system will construct a 3D region. The detection algorithm then determines if a person has entered an off-limits area based on predefined closing times. This solution enhances security by ensuring that intrusions are only flagged after business hours for each specific zone.  
# SETUP GUIDE  
## I. Prerequisites  
Ensure that your system meets the following requirements:  
•	Python 3.8+  
•	CUDA-compatible GPU (Optional for better performance)  
•	Required dependencies: All required packages are listed in requirements.txt file that located in folder code.    
## II. Installation Steps  
### 1. Clone the Repository  
• **Clone our repository**: `git clone https://github.com/nhnphu5525/intruder_detection/`  
• **Navigate to repository folder**: `cd intruder_detection`  
### 2. Install Dependencies  
• **Install necessary packages**: `pip install -r requirements.txt`
### 3. Ensure PyTorch Supports CUDA (Optional for GPU Acceleration)  
```
import torch  
print(torch.cuda.is_available())  # Should return True if GPU is available
```
### 4. Run the Application  
• **Run app command**: `python main.py`
# User Guide  
This guide provides an overview of how to set up and use the Intrusion 3D Detection system effectively. For further customization, modify main.py and adjust the detection logic as needed.  
## 1. Selecting an Intrusion Zone  
•	When the application starts, a window will appear displaying the video feed.  
•	Left-click to select points defining the bottom of the intrusion zone.  
•	Right-click to remove the last selected point if needed.  
•	Press C to confirm the zone and create a corresponding 3D intrusion region.   
## 2. Assigning Operating Hours  
•	Each intrusion zone has a unique opening and closing time.  
•	The application checks if an intrusion occurs after closing hours.  
•	Zones and their assigned operating hours will be displayed on the screen.  
## 3. Detecting Intrusions  
•	Once a person enters a zone after closing time, an alert is triggered.  
•	The detection algorithm ensures intrusions are flagged only when both foot position and head height fall within the intrusion area.  
## 4. Exiting the Application
•	Press Q at any time to quit the application.    



