#  Facial Expression Recognition using Vision Transformers (ViT)

This project explores the application of ViT for recognizing facial expressions. The code leverages the power of ViT to extract meaningful features from facial images, enabling accurate emotion classification.

## Getting Started

1. Clone the Repository:
   Use Git to clone this repository to your local machine:

   ```bash
   git clone https://github.com/sairam-penjarla/facial-expression-recognition
   ```

2. Install Dependencies:
   Navigate to the project directory and install the required libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. Download Dataset:
   Important Note: This project does not include the dataset due to size and licensing restrictions. However, you can download the AffectNet training data from Kaggle:

   [https://www.kaggle.com/datasets/noamsegal/affectnet-training-data](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data)

   Once downloaded, extract the dataset files and place them in a designated `dataset` folder within your project directory. The code will assume a specific directory structure for the dataset. You may need to adjust the paths in `main.py` if your structure differs.

4. Run the Script:
   Navigate to the `scripts` directory within the project and execute the main script:

   ```bash
   cd scripts
   python main.py
   ```

This will initiate the facial expression recognition process using the ViT model. The script will perform the following:

   - Load the pre-trained ViT model
   - Load the downloaded facial expression dataset
   - Preprocess the images (resizing, normalization)
   - Train the ViT model on the dataset
   - Evaluate the model's performance
   - Predict facial expressions for new images (optional)

## Theory and Applications

Facial expression recognition (FER) is a vital field in computer vision with numerous applications:

- Human-Computer Interaction (HCI): Systems can adapt to user emotions in real-time, enhancing user experience in applications, games, and virtual reality.
- Affective Computing: Analyze user responses to content, leading to personalized recommendations and improved marketing strategies.
- Surveillance and Security: Automatic detection of suspicious or aggressive behavior in public spaces or border control.
- Medical Diagnosis: Assist healthcare professionals in identifying emotional signs of depression, anxiety, or pain.
- Education and Learning: Systems can monitor student engagement and emotional well-being, tailoring instruction accordingly.

## Vision Transformers (ViTs):

ViTs, introduced in 2020 by Dosovitskiy et al., represent a significant advancement in computer vision. They break away from traditional Convolutional Neural Networks (CNNs) by employing a pure attention mechanism for image classification:

- Image Splitting: An input image is divided into patches.
- Patch Embedding: Each patch is converted into a vector representation using a linear transformation.
- Positional Encoding: Relative positions of patches are encoded to capture spatial information.
- Transformer Encoder: A series of encoder layers process the embedded patches, learning relationships between them.
- Classification: The final output is fed into a classifier head for emotion prediction.

## ViTs offer advantages over CNNs:

- Global Context Awareness: ViTs excel at capturing long-range dependencies and global context within an image, crucial for tasks like FER where subtle facial expressions hold significance.
- Flexibility: ViTs are easily adaptable to different input sizes and can be pre-trained on large image datasets for transfer learning to specific tasks like FER.

### Disclaimer

- Modifying the code or dataset structure might necessitate changes to the script.
- Consider potential biases in datasets and the limitations of FER systems.

## Future Enhancements

Explore fine-tuning the ViT model with different hyperparameters.
- Integrate data augmentation techniques to improve model robustness.
- Experiment with pre-trained ViT models specifically designed for facial recognition.
- Visualize the learned attention weights to understand how the model attends to critical facial regions.
- Explore advanced techniques like ensemble learning or multi-modal fusion (combining facial expressions with other modalities like speech) for enhanced accuracy.