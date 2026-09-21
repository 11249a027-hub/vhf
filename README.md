# vhf
# VHF Signature Fraud Detection

## About the Project

VHF Signature Fraud Detection is an AI-based project developed to verify handwritten signatures and identify possible forged signatures.

The main idea of the project is to compare a newly submitted signature with genuine signature samples that were previously enrolled for the respective person. The system uses a Siamese Neural Network to learn signature patterns and determine whether the submitted signature matches the enrolled signatures.

The project is developed with the goal of exploring how AI and image processing can be used for signature verification in financial and banking-related applications.

## What the System Does

The system allows us to:

- Register a person and store their details.
- Enroll multiple genuine signature samples.
- Upload a signature for verification.
- Compare the submitted signature with the enrolled signatures.
- Generate a similarity score.
- Classify the signature as genuine or forged.
- Maintain verification history.

## How It Works

First, genuine signatures are enrolled for a person.

When a new signature is submitted, the system processes the signature image and passes it through the trained Siamese Neural Network. The model generates a representation of the signature, which is then compared with the enrolled genuine signatures.

Based on the similarity between the signatures, the system provides the verification result.

## Technologies Used

- Python
- PyTorch
- FastAPI
- JavaScript
- HTML
- CSS
- OpenCV
- SQLite / SQL Database
- Siamese Neural Network

## Dataset

The model was trained and evaluated using the CEDAR Signature Dataset, which contains genuine and forged handwritten signatures.

The complete dataset is not included in this repository because of its large size.

## Project Structure

The project contains separate sections for the backend, frontend, machine learning model, and signature data.

```text
backend/       - Backend and API
frontend/      - User interface
ml/            - Machine learning and signature processing
ml_models/     - Trained model
signatures/    - Signature-related files
