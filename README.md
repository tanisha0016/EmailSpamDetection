# EmailSpamDetection
Here's a `README.md` file for your spam detection project:

```markdown
# Spam Detection Using Logistic Regression

This project demonstrates the use of Logistic Regression to classify emails as spam or ham (non-spam) using the TF-IDF vectorization technique.

## Dataset

The dataset used for this project is a collection of SMS messages that have been classified as either "spam" or "ham".

## Files

- `spam.csv`: The dataset file containing the SMS messages and their labels.
- `spam_detection.ipynb`: The Jupyter Notebook containing the code for the project.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/your-repo-name.git
   ```
2. Navigate to the project directory:
   ```sh
   cd your-repo-name
   ```
3. Install the required libraries:
   ```sh
   pip install numpy pandas scikit-learn
   ```

## Usage

1. Open the Jupyter Notebook:
   ```sh
   jupyter notebook spam_detection.ipynb
   ```
2. Run the cells in the notebook to execute the code.

## Code Overview

### Import Libraries
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### Load and Prepare Data
```python
raw_mail_data = pd.read_csv('/content/spam.csv', encoding='ISO-8859-1')
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
mail_data.loc[mail_data['v1'] == 'spam', 'v1',] = 0
mail_data.loc[mail_data['v1'] == 'ham', 'v1',] = 1
X = mail_data['v2']
Y = mail_data['v1']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
X_train = X_train.astype(str)
X_test = X_test.astype(str)
```

### Feature Extraction
```python
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
```

### Model Training
```python
model = LogisticRegression()
model.fit(X_train_features, Y_train)
```

### Model Evaluation
```python
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data:', accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data:', accuracy_on_test_data)
```

### Predictive System
```python
input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]
input_mail_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_mail_features)
print(prediction)
if (prediction[0] == 1):
    print('Ham mail')
else:
    print('Spam mail')
```

## Results

- Accuracy on training data: 97.33%
- Accuracy on test data: 97.31%

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

```

You can copy this content into a `README.md` file in your repository. It provides an overview of the project, including the dataset, dependencies, usage, and a code overview.
