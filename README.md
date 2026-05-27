# Brain Tumor Detection using MobileNetV2

## Project Overview

This project is developed for the CSC4101 Artificial Intelligence Semester Project.

The system detects different types of brain tumors from MRI images using a pretrained MobileNetV2 deep learning model.

The project includes:
- Model Training
- Batch Inference
- Flask API
- Docker Deployment

---

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- Flask
- Docker

---

## Dataset

Brain MRI dataset containing 4 classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor

---

## Model Used

Pretrained Model:
- MobileNetV2

Transfer Learning was used for image classification.

---

## Project Structure

```bash
AI-BrainTumor-Detection/
│
├── train.py
├── inference.py
├── app.py
├── Dockerfile
├── requirements.txt
├── BrainTumorTraining.ipynb
├── report.docx
├── README.md
├── brain_model.h5
```

---

## Training

Run training script:

```bash
python train.py
```

---

## Batch Inference

Run inference script:

```bash
python inference.py
```

Example Output:

```bash
Te-aug-me_7.jpg --> meningioma (97.54%)
Te-gl_6.jpg --> glioma (95.22%)
Te-pi_14.jpg --> pituitary (98.11%)
Te-pi_6.jpg --> pituitary (96.80%)
```

---

## Flask API

Run Flask server:

```bash
python app.py
```

API will run on:

```bash
http://127.0.0.1:5000
```

---

## Docker Commands

Build Docker image:

```bash
docker build -t brain-tumor-api .
```

Run Docker container:

```bash
docker run -p 5000:5000 brain-tumor-api
```

---

## Evaluation Metrics

- Accuracy: Add your final accuracy here
- Loss: Add your final loss here

Example:

```bash
Test Accuracy: 95.14%
Test Loss: 0.1623
```

---

## GitHub Repository


```bash
https://github.com/SymbolPamnani/AI-Brain-Tumor-Detection-Project```

---

## Docker Hub Link

```bash
https://hub.docker.com/r/symbolpamnani/brain-tumor-api
```

---
