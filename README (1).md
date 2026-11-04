# Weather Classification System

A Streamlit-based machine learning application that explores a weather dataset, visualizes weather attributes, evaluates a classification model, and predicts the weather type from user-provided inputs.

## Features

- Upload a weather dataset in CSV format.
- Preview the uploaded data and inspect its dimensions.
- View missing-value counts and descriptive statistics.
- Explore numeric features with histograms.
- View a correlation heatmap for numeric columns.
- Encode categorical features for machine learning.
- Train and evaluate a Random Forest classifier.
- Display model accuracy and a classification report.
- Enter weather conditions and predict the weather category.

## Project Structure

```text
weather/
├── app.py
├── weather_classification_data.csv
├── best_svm_weather_model.pkl
├── label_encoders.pkl
├── scaler (1).pkl
├── selector.pkl
└── requrirements.txt
```

**Note:** The current `app.py` trains a `RandomForestClassifier` when a CSV file is uploaded. The saved SVM model, scaler, selector, and encoder pickle files are not loaded by the current application code. They may belong to a separate model-training workflow.

## Dataset

The included `weather_classification_data.csv` dataset contains 13,200 rows and 11 columns:

| Column | Description |
|---|---|
| `Temperature` | Temperature reading |
| `Humidity` | Humidity level |
| `Wind Speed` | Wind speed |
| `Precipitation (%)` | Precipitation percentage |
| `Cloud Cover` | Cloud-cover category |
| `Atmospheric Pressure` | Atmospheric pressure |
| `UV Index` | UV index |
| `Season` | Season |
| `Visibility (km)` | Visibility distance in kilometres |
| `Location` | Location category |
| `Weather Type` | Target class to predict |

The target column required by the application is **`Weather Type`**.

## Requirements

- Python 3.9 or later
- pip

Python packages:

- streamlit
- pandas
- matplotlib
- seaborn
- scikit-learn

## Installation

1. Download or extract this project.
2. Open a terminal in the project directory.
3. (Recommended) Create and activate a virtual environment:

   **Windows**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   **macOS / Linux**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install the required packages:

   ```bash
   python -m pip install --upgrade pip
   pip install streamlit pandas matplotlib seaborn scikit-learn
   ```

   The bundled dependency file is named `requrirements.txt` (misspelled) and is encoded as UTF-16. The commands above avoid potential installation issues with that file. If you want to use a requirements file, create a UTF-8 file named `requirements.txt` containing the package names listed above.

## Run the Application

From the project directory, run:

```bash
streamlit run app.py
```

Streamlit will print a local URL (normally `http://localhost:8501`). Open that URL in your browser.

## How to Use

1. Launch the application.
2. Upload a CSV file with the same feature structure as the included dataset.
3. Review the dataset preview, shape, missing values, and statistical summary.
4. Select a numeric feature to view its distribution.
5. Review the correlation heatmap.
6. Check the Random Forest model's accuracy and classification report.
7. Use the input controls under **Predict Weather Type** to enter weather conditions.
8. Click **Predict** to display the predicted weather category.

## Input Data Requirements

- The CSV must contain a column named exactly `Weather Type`.
- It should contain the feature columns used by the model, such as temperature, humidity, wind speed, cloud cover, season, and location.
- Use consistent column names and compatible data types.
- The current application does not include explicit handling for missing values or unseen categories. Clean and validate uploaded data before using it.

## Model Workflow

The current application follows this workflow:

1. Read the uploaded CSV file with pandas.
2. Encode columns with text/object data using `LabelEncoder`.
3. Separate the features (`X`) from the target (`Weather Type`).
4. Split the data into training and testing sets (80% / 20%).
5. Train a `RandomForestClassifier` with `random_state=42`.
6. Calculate accuracy and generate a classification report.
7. Collect user inputs and predict a weather class.

The model is retrained whenever a CSV file is uploaded; the application does not currently load the bundled `.pkl` model files.

## Troubleshooting

- **`streamlit` is not recognized:** Activate your virtual environment and run `python -m pip install streamlit`.
- **CSV upload fails:** Confirm that the file is a valid CSV and uses the expected column names.
- **`Weather Type` column error:** Ensure the target column exists and is spelled exactly `Weather Type`.
- **Prediction errors:** Check for missing values, unexpected categories, and feature columns that differ from the training data.
- **Dependency installation errors:** Install the packages individually using the installation command above.

## Future Improvements

- Add data validation and missing-value handling.
- Persist the trained model rather than retraining on every upload.
- Load and use the intended saved SVM model and its preprocessing artifacts, if that is the desired production workflow.
- Add confusion matrix and per-class performance visualizations.
- Separate training, preprocessing, and prediction logic into reusable modules.
- Add tests and clearer error messages for invalid datasets.

## License

Add the applicable license and attribution details before distributing this project.
