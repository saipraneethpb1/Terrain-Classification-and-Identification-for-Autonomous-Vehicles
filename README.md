# Terrain Classification and Identification for Autonomous Vehicles🏔️

This project focuses on enhancing the safety and adaptability of autonomous vehicles by enabling reliable identification and classification of off-road terrains. By leveraging the YOLOv8 instance segmentation model, the project achieves accurate terrain detection, supporting autonomous vehicles in making informed navigation decisions.

The model was trained using the CAT-CaVS Traversability Dataset, which includes diverse off-road scenarios. Key features include the use of advanced convolutional neural network (CNN) architectures, real-time image segmentation, and precise delineation of drivable regions.

This research lays the groundwork for integrating terrain analysis with practical hardware implementations, paving the way for further advancements in autonomous vehicle technologies.

## Overview
Off-road terrain plays a critical role in determining the performance and safety of autonomous vehicles. This project uses YOLOv8, a state-of-the-art CNN architecture, for instance segmentation to classify drivable and non-drivable regions in complex terrains. The model is trained on annotated off-road images and demonstrates high accuracy in detecting drivable regions.

## Features
- **Instance Segmentation:** Differentiates drivable regions from obstacles such as rocks, trees, and rough terrain.
- **Real-Time Analysis:** Processes images to identify paths in off-road scenarios with high precision.
- **Preprocessing Pipeline:** Includes resizing, augmentation, and annotation refinement for robust model training.
- **Scalable Framework:** Can be extended to include path planning and obstacle avoidance.

## Data Collecting and Cleaning

**Dataset:**

The Dataset used for the Experiment to identify the Terrain is Cat: CAVS Traversability Dataset which consists 
of 3.45k Off-Road Terrain images which were collected by Mississippi State University and is stated in. The 
data collection was performed near HPCC (High Performance Company Collaboratory) of Mississippi State 
University. The data is collected from three trails i.e., main trail with a length of 0.64 km , powerline trail with 
a length of 0.82 km, brownfield trail with a length of 0.21 km . The images were collected considering the 
required light exposure with different filters compatible with Sekonix SF3325-100 camara model.

source: https://www.cavs.msstate.edu/resources/autonomous_dataset.php#:~:text=CaT%3A%20CAVS%20Traversibility%20Dataset&text=This%20dataset%20contains%20a%20set,to%20drive%20through%20the%20terrain

**Data Cleaning Process**

 In the final dataset, we retained only the most relevant features for the project.

    1. Removed Duplicative Copies:
      - Removed similar images to train diverse off-road scenarios for better accuracy
    2. Image Resize:
      - The image dimensions were resized to 640x640 pixels to ensure uniformity across all images.  
    3. Data Augmentation:
      -  Applied color adjustments such as brightness, contrast, and saturation changes, along with geometric transformations like rotation and flipping, to enhance dataset diversity and improve model generalization.
This process ensured a clean and concise dataset focused solely on features relevant to the research.

## Annotation Refinement:

- **Platform:** Used Roboflow for efficient and user-friendly manual labeling.
- **Focus:** Labeled drivable regions in images, distinguishing them from obstacles and non-drivable terrain.
- **Workflow:**
  - Uploaded raw images to Roboflow.
  - Outlined drivable regions and assigned specific color-coded masks.
  - Performed quality checks to ensure accuracy and consistency.
- **Refinement:** Iterative corrections were made to improve annotation precision.
- **Dataset:** Selected 1.24k high-quality annotated images from the 3.45k dataset for training.
- **Outcome:** Enhanced data quality improved the YOLOv8 model's ability to accurately detect drivable regions.
- **Advantages:**
  - Ensured precise and clean data for training.
  - Improved segmentation accuracy and model performance.

## Model Development

**Training Parameters**

- **Model Architecture:** YOLOv8 instance segmentation model, a state-of-the-art Convolutional Neural Network (CNN) for object detection and segmentation.
- **Training Configuration:**
  - Epochs: 100
  - Image Size: 640 pixels
  - Batch Size: 16
  - Loss Functions: Box loss, segmentation loss, and class loss were monitored during training for both training and validation data.
  - Feature extraction across 21 stages for refined segmentation performance.

**Validation Parameters**

- Validation data constituted 8% of the total dataset to evaluate model generalization.
- Performance metrics such as precision, recall, and F1-score were monitored to measure the effectiveness of the trained model.
- The confusion matrix was analyzed to identify misclassifications and refine training where necessary.

**Testing Parameters**

- Testing data formed 4% of the dataset and was used to validate the model’s real-world applicability.
- Predicted outputs were visually compared with ground truth annotations, showcasing the model's ability to segment drivable regions and differentiate obstacles.


## Feature extraction

Feature extraction is a crucial step where the model processes input images through its backbone network to identify and extract essential patterns, such as edges, textures, and object-specific details. 
- **conv_features:** It refer to the feature maps generated by convolutional layers in a neural network, capturing spatial hierarchies and patterns like edges, textures, and shapes from input data.(stage0,stage1,stage3,stage5,stage7,stage16,stage19)
  
  ![stage0_Conv_features](https://github.com/user-attachments/assets/bb4170d2-9e4b-4393-8604-fe139dc83e88)
  
- **c2f_features:** These are features generated by a C2f (Cross-Stage Partial Fusion) module, which improves computational efficiency and gradient flow by splitting feature maps into two groups, processing one while shortcutting the other, then merging them.(stage2,stage4,stage6,stage8,stage12,stage15,stage18,stage21)
  
  ![stage2_C2f_features](https://github.com/user-attachments/assets/2f88ce54-2a16-4397-bf55-d2bb48b33191)
  
- **sppf_features:** These features are derived from a SPPF (Spatial Pyramid Pooling-Fast) module, which pools feature maps at multiple scales to enhance the receptive field and capture contextual information at different resolutions.(stage9)
  
  ![stage9_SPPF_features](https://github.com/user-attachments/assets/ad7c68f5-d629-4358-aeb1-64910737fa02)
  
- **up_sample_features:** These are features obtained after applying an up-sampling operation, where spatial resolution is increased (e.g., through interpolation or transposed convolutions) to restore or match the size of a previous layer's feature map.(stage10,stage13)
  
  ![stage10_Upsample_features](https://github.com/user-attachments/assets/58e40b41-f0dc-4998-8ab1-31509ed42916)
  
- **concat_features:** These are features produced by concatenating multiple feature maps along the channel dimension, often used to merge information from different layers or paths in a network for richer representation.(stage11,stage14,stage17,stage20)
  
  ![stage11_Concat_features](https://github.com/user-attachments/assets/d8d3819d-d56a-49ac-8552-48f3b7ae81f7)

## Result

**Performance Metrics:**

- **Precision:** It measures the proportion of true positive predictions out of all positive predictions, indicating the accuracy of the model's positive classifications.
- **Recall:** It measures the proportion of true positive predictions out of all actual positive cases, indicating the model's ability to identify all relevant instances.
- **F1-Score:** Harmonic mean of precision and recall, providing a balanced measure of a model's accuracy that considers both false positives and false negatives.
- **mAP@50 (Mean Average Precision at IoU 0.50):** This metric evaluates the detection model's performance by calculating the average precision (AP) for all classes at a single Intersection over Union (IoU) threshold of 0.50, which is considered a lenient criterion for overlap between predicted and ground truth bounding boxes.
- **mAP@50-95 (Mean Average Precision at IoU 0.50 to 0.95):** This metric provides a more comprehensive evaluation by averaging AP over multiple IoU thresholds ranging from 0.50 to 0.95 (in steps of 0.05), offering a stricter and more detailed measure of the model's detection accuracy across varying overlap levels.

**Loss Function:**

- **Box loss (box_loss):** It measures the discrepancy between the predicted bounding boxes and the ground truth bounding boxes, typically using metrics like Smooth L1 loss or IoU-based loss to optimize the model's localization accuracy.
- **Segmentation Loss (seg_loss):** It quantifies the difference between the predicted and ground truth segmentation masks, commonly using metrics like Cross-Entropy Loss or Dice Loss, to optimize pixel-wise classification accuracy.
- **Classification Loss (cls_loss):** It measures the error in predicting class labels for objects, typically using Cross-Entropy Loss, Focal Loss, or other variations to enhance the model's accuracy in distinguishing between categories.
- **Distribution Focal Loss (dfl_loss):** It is used in object detection to refine bounding box regression by learning the distribution of precise locations, improving the accuracy of boundary predictions for objects.

![results](https://github.com/user-attachments/assets/10d85de8-efe1-4226-95be-79f24aa89bb9)

![image](https://github.com/user-attachments/assets/a57ddfbb-ba42-4261-b252-a5b150c3af16)  ![image](https://github.com/user-attachments/assets/8dd977d0-3f6e-451e-a20b-91e657b3898c)


## Conclusion

This project presents a comprehensive approach to off-road terrain identification and analysis, utilizing the YOLOv8 instance segmentation architecture. By training the model on the CAT-CaVS Traversability Dataset, which includes diverse off-road terrain images, the system achieves high accuracy in detecting drivable regions and differentiating them from obstacles such as trees, rocks, and other environmental features. The YOLOv8 model's advanced capabilities in instance segmentation allow it to deliver refined and precise results, making it well-suited for real-world applications in autonomous vehicles.

The study emphasizes the importance of computational techniques in enabling autonomous vehicles to adapt to challenging terrains, a critical factor for ensuring safety and efficiency in navigation. The current implementation focuses on computational aspects, providing a robust foundation for further research into integrating hardware such as sensors, accelerometers, and LIDAR for real-time terrain analysis.

Future work will aim to develop a more generalized and sophisticated hardware model capable of handling diverse terrains while dynamically adjusting vehicle maneuverability. The trained model can also be extended to incorporate additional functionalities like path planning, object detection, and GPS integration, moving closer to the realization of fully autonomous vehicles. This research motivates ongoing advancements in autonomous navigation systems, contributing to the broader adoption of Industry 4.0 technologies and improving the reliability and safety of self-driving vehicles across various environments.


## Project Notebook

The notebook seg.ipynb contains all the detailed steps for this project.

You can view or download the notebook directly from this repository:
 - 📒 [Jupyter Notebook](https://github.com/mani9kanta3/Terrain_Segmentation/blob/main/seg.ipynb)

## Future Directions

The project provides a foundation for off-road terrain identification and analysis using advanced computer vision techniques. Future work can expand upon this by focusing on the following directions:

- **Hardware Integration:** Implement the trained terrain identification model on real-time hardware platforms, such as autonomous ground vehicles equipped with advanced sensors (e.g., LIDAR, IMU, accelerometers) for real-world testing and validation.

- **Dataset Expansion:** Collect more diverse and high-resolution datasets encompassing various terrain types and environmental conditions. Integrate advanced camera modules to capture better quality data for improved model training.

- **Model Enhancement:** Explore integrating the YOLOv8 model with other deep learning architectures to improve segmentation accuracy and computational efficiency. This includes experimenting with hybrid models for better adaptability to complex terrains.

- **Path Planning and Obstacle Avoidance:** Extend the current work to include path planning and obstacle avoidance systems by combining terrain detection with object recognition, GPS, and real-time navigation.

- **Environmental Adaptability:** Develop models that can adapt to changing environmental conditions, such as varying light, weather, and visibility, for more robust terrain analysis in diverse scenarios.

- **Energy Efficiency:** Optimize the model for deployment on edge devices with limited computational power, making it feasible for low-cost autonomous vehicle systems.

- **Real-Time Applications:** Extend the project to support real-time applications by reducing model inference time and integrating it with efficient data pipelines.

These directions aim to advance the field of autonomous navigation by enhancing the capabilities and reliability of terrain identification and analysis systems in off-road and challenging environments.

## References

1. Dataset Utilized: S. Sharma et al., "CaT: CAVS Traversability Dataset for Off-Road Autonomous Driving," IEEE Access, vol. 10, pp. 24759–24768, 2022, doi: 10.1109/ACCESS.2022.3154419.

2. Model Architecture: J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection." Available online: http://pjreddie.com/yolo/.

3. Radar Imaging: S. Royo and M. Ballesta-Garcia, "An overview of lidar imaging systems for autonomous vehicles," Applied Sciences (Switzerland), vol. 9, no. 19, Oct. 2019, doi: 10.3390/app9194093.

4. Semantic Segmentation Techniques: D. Jiang et al., "Semantic segmentation for multiscale target based on object recognition using the improved Faster-RCNN model," Future Generation Computer Systems, vol. 123, pp. 94–104, Oct. 2021, doi: 10.1016/j.future.2021.04.019.

5. NuScenes Dataset: H. Caesar et al., "nuScenes: A multimodal dataset for autonomous driving," Mar. 2019. Available online: http://arxiv.org/abs/1903.11027.
