# Speech-Emotion-Recognition
Speech Emotion Recognition (SER) with Deep Learning 🗣️🧠
🚀 Project Overview
This repository presents a robust Speech Emotion Recognition (SER) system, designed to classify human emotional states from spoken audio. The project employs advanced deep learning architectures, specifically Artificial Neural Networks (ANN) and two distinct types of Convolutional Neural Networks (CNNs), to analyze intricate acoustic features extracted from speech. Its primary objective is to accurately identify a range of emotions, including neutral, happy, sad, angry, fearful, disgusted, and surprised, providing valuable insights into affective computing from voice.

✨ Key Features
Multi-Dataset Aggregation: Consolidates and processes speech data from four prominent emotional speech datasets (RAVDESS, TESS, CREMA-D, SAVEE) to build a diverse and generalized training corpus.

Advanced Acoustic Feature Engineering: Utilizes multiple crucial acoustic features:

Mel-Frequency Cepstral Coefficients (MFCCs): Captures the timbre and spectral characteristics of speech.

Chroma Features: Represents the twelve different pitch classes.

Mel Spectrogram: Provides a time-frequency representation of the audio signal on a Mel scale.
Features are processed (e.g., mean and variance across frames) and concatenated to form a comprehensive input vector.

Diverse Deep Learning Architectures: Implements and evaluates multiple neural network models for robust emotion classification:

Artificial Neural Network (ANN): A foundational multi-layer perceptron for baseline emotion classification.

1D Convolutional Neural Network (1D CNN): Specifically tailored for processing sequential audio features, capturing temporal patterns effectively.

2D Convolutional Neural Network (2D CNN): Designed to process spectrogram-like features (e.g., Mel spectrograms or MFCC images) by treating them as images, leveraging spatial patterns.

Comprehensive Data Preprocessing Pipeline: Handles audio loading, feature extraction, padding/truncating features to a uniform length, data splitting (train_test_split), and one-hot encoding of emotional labels.

Interactive Jupyter Notebook: The entire process—from data loading and feature engineering to model training, evaluation, and real-time prediction—is thoroughly documented and executable within the Speech_Emotion_Recognition.ipynb notebook.

📊 Datasets
The model is trained on a combined dataset to enhance its robustness and generalization capabilities. You will need to download these datasets manually due to their large size.

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song): Features 24 professional actors (12 male, 12 female) vocalizing emotional states.

TESS (Toronto Emotional Speech Set): Composed of 200 target words spoken by two actresses, portraying various emotions.

CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset): Includes 7,442 clips from 91 actors with diverse ethnic backgrounds, speaking sentences in 6 different emotions.

SAVEE (Surrey Audio-Visual Expressed Emotion): Contains emotional utterances from 4 male actors.

Data Organization: After downloading, please place the unzipped dataset folders (e.g., RAVDESS, TESS, etc.) inside a root Datasets/ directory within this project's cloned folder. The notebook is configured to read from this structure.

🛠️ Technologies & Libraries
This project is developed in Python and leverages the following core libraries:

Python 3.x

TensorFlow / Keras: The primary framework for building and training the deep learning models.

Librosa: Indispensable for audio loading, signal processing, and comprehensive feature extraction (MFCCs, Chroma, Mel Spectrograms).

SoundFile: Used for efficient reading and writing of audio files (.wav, etc.).

NumPy: Essential for high-performance numerical operations and array manipulation.

Pandas: For structured data handling, especially for managing dataset metadata and labels.

Scikit-learn (sklearn): Utilized for data splitting (train_test_split), label encoding (LabelEncoder), and model evaluation metrics.

Matplotlib: For creating static, interactive, and animated visualizations.

Jupyter Notebook: The interactive environment for running and exploring the code.

⚙️ Installation & Setup
To get this project running on your local machine, follow these steps:

Clone the Repository:

Bash

git clone https://github.com/Subhradip-2003/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition

tensorflow
librosa
soundfile
numpy
pandas
scikit-learn
matplotlib
jupyter
Then, install them using pip:

Bash

pip install -r requirements.txt
Download and Place Datasets:
As mentioned in the Datasets section, download the RAVDESS, TESS, CREMA-D, and SAVEE datasets. Create a folder named Datasets in your project's root directory, and place the unzipped contents of each dataset into this Datasets folder.

🏃‍♀️ Usage
Launch Jupyter Notebook:
From your project's root directory in the terminal (with your virtual environment activated), execute:

Bash

jupyter notebook
Open the Project Notebook:
Your web browser will open, displaying the Jupyter interface. Navigate to and open the Speech_Emotion_Recognition.ipynb file.

Execute Cells:
Run all cells sequentially within the notebook. The notebook will guide you through:

Loading audio data and their corresponding emotional labels.

Extracting MFCC, Chroma, and Mel features.

Preprocessing the features and labels for model training.

Defining, compiling, and training the ANN, 1D CNN, and 2D CNN models.

Evaluating the trained models' performance on unseen data.

Demonstrating how to use the models for real-time (or single-sample) emotion prediction.

📈 Model Performance & Evaluation
The notebook includes comprehensive sections for evaluating the trained models. You'll observe:

Training and Validation Loss/Accuracy Curves: Plots illustrating each model's learning progress over epochs.

Comparative Analysis: The notebook allows for a comparison of performance metrics (accuracy, loss) across the ANN, 1D CNN, and 2D CNN architectures, highlighting their respective strengths for this SER task.

Test Set Accuracy: The final accuracy achieved by each model on the unseen test dataset.

The performance analysis will help demonstrate the effectiveness and comparative advantages of different deep learning approaches for Speech Emotion Recognition.


📧 Contact
For any questions, feedback, or collaborations, feel free to connect:

Subhradip Bhattacharyya

LinkedIn: www.linkedin.com/in/subhradip2003

Email: subhradipbhattacharyya290@gmail.com
