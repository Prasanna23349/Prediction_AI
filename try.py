import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Load and preprocess the data
df = pd.read_csv('dataset.csv')
df.dropna(subset=['Description', 'Class'], inplace=True)

# Drop rows where 'Class' is 'A', '200', or 'B'
values_to_drop = ['A', '200', 'B']
df = df[~df['Class'].isin(values_to_drop)]

X = df['Description']
y = df['Class']

# Text preprocessing
voc_size = 5000
ps = PorterStemmer()
corpus = []

for i in range(len(X)):
    review = re.sub('[^a-zA-Z]', ' ', X.iloc[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# One-hot encode the corpus
onehot_repr = [one_hot(words, voc_size) for words in corpus]

# Pad the sequences
sent_length = 20
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)

# Encode labels as integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# One-hot encode the labels
y = to_categorical(y, num_classes=len(np.unique(y)))

# Convert to numpy arrays
X_final = np.array(embedded_docs)
y_final = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

# Build the model
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(46, activation='softmax'))  # Number of classes as output
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
print(model.summary())

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
