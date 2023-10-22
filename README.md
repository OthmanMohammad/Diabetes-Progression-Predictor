# Diabetes Progression Predictor

This repository exemplifies a robust ML workflow, leveraging MLflow for experiment tracking, Docker for containerization, TensorFlow Serving for model deployment, and GitHub Actions for CI/CD. It embodies a comprehensive system designed to predict diabetes progression using advanced machine learning paradigms.

## Table of Contents
- [Overview](#overview)
- [Using MLflow for Model Tracking](#using-mlflow-for-model-tracking)
- [Dockerization](#dockerization)
  - [Training Dockerfile](#training-dockerfile)
  - [Serving Dockerfile](#serving-dockerfile)
- [GitHub Actions Workflows](#github-actions-workflows)
- [TensorFlow Serving](#tensorflow-serving)
- [Usage](#usage)

## Overview
The primary objective of this project is to demonstrate a modern MLOps pipeline by predicting diabetes progression using specific input features. This pipeline incorporates model training, tracking with MLflow, containerization via Docker, serving with TensorFlow Serving, and automating workflows with GitHub Actions.

### Using MLflow for Model Tracking
MLflow is integrated into the training pipeline to keep track of different model runs. It logs metrics, parameters, and the trained model artifacts. This ensures reproducibility and easy comparison between different runs.

Steps to use MLflow:

1. Run your training script.
2. Start the MLflow tracking UI:
   ```bash
   mlflow ui
3. Navigate to the MLflow dashboard by visiting http://127.0.0.1:5000.

### Dockerization
We utilize Docker for packaging our training and serving components.

#### Training Dockerfile
This Dockerfile is set up to run the training script. It ensures that all dependencies are met and provides an isolated environment to run the training.

To build the training image:

```bash
docker build . --file Dockerfile.train --tag [YOUR_TAG_NAME]
```
## Serving Dockerfile

This Dockerfile packages the TensorFlow Serving setup with our trained model. It facilitates serving the model for predictions.

To build the serving image:

```bash
docker build . --file Dockerfile.serving --tag [YOUR_SERVING_TAG_NAME]
```
## GitHub Actions Workflows

We've set up three main GitHub Actions workflows:

1. **Docker Image CI/CD for Training**: This workflow automatically builds the training Docker image and pushes it to Docker Hub upon any push to the repository.

2. **Python Tests**: It ensures that the codebase remains robust and functional by running a suite of tests on every push.

3. **TensorFlow Serving CI/CD**: This builds and pushes the TensorFlow Serving Docker image to Docker Hub on every push.

## TensorFlow Serving

We employ TensorFlow Serving for efficient and scalable serving of our model. Once the Docker image for TensorFlow Serving is built, you can run it to serve your model on a specific port.

## Usage

1. **Train the Model**: This can be done by running the training script directly or through the training Docker container.

2. **Serve the Model**: Run the TensorFlow Serving Docker container to start serving the model.

3. **Make Predictions**: With the model being served, send a POST request with the input features to get the predicted progression.



