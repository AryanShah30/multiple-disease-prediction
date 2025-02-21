# Multiple Disease Prediction System

[View Disease Risk Assessment](https://multiple-diseases-prediction-system.streamlit.app/)

## Overview

This project is a web application that utilizes machine learning models to predict the likelihood of various health conditions based on user-provided data. The system covers predictions for Diabetes, Heart Disease, Parkinson's Disease, and Breast Cancer. Built using Streamlit, the application offers a user-friendly interface for inputting health metrics and receiving predictions.

## Features

- **Diabetes Prediction**: Assesses diabetes risk using health metrics
- **Heart Disease Prediction**: Evaluates heart disease risk based on health indicators
- **Parkinson's Disease Prediction**: Determines Parkinson's disease probability using specific parameters
- **Breast Cancer Prediction**: Classifies breast cancer as benign or malignant based on user input

## Installation

### Requirements

- Python 3.x
- Streamlit (~=1.36.0)
- NumPy (~=2.0.0)

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone 
   cd 
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Access the application at `http://localhost:8501` in your web browser.

## Usage

1. Select a disease prediction model from the sidebar.
2. Input required health metrics in the provided fields.
3. Click the prediction button to receive results.
4. View the prediction indicating the likelihood of having the selected disease.

## Technical Details

- The application uses pre-trained machine learning models for each disease prediction.
- Models are loaded from saved files using the `pickle` library.
- Streamlit is used for creating the web interface and handling user interactions.
- NumPy is utilized for numerical computations and data processing.

## Contributing
Contributions to enhance the analysis or extend the project are welcome. Please feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For inquiries, suggestions, or feedback, please reach out to: [AryanShah30](https://github.com/AryanShah30)
