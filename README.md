# LSTM-experiments

This repository contains two projects showcasing the use of LSTM (Long Short-Term Memory) neural networks to learn and predict patterns from sequences. Both projects are implemented in Python using TensorFlow and Keras.

---

## Project 1: Predicting the Fibonacci Sequence

This project demonstrates how to use an LSTM model to learn the Fibonacci sequence and predict the next number in the series.

### Key Steps:
1. Generate a Fibonacci sequence.
2. Prepare the dataset using sliding windows.
3. Build and train an LSTM model.
4. Use the model to predict the next value in the sequence.

### Model Architecture:
- Input: shape `(time_steps, 1)`
- LSTM layer with 50 units and ReLU activation
- Dense layer with 1 output
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam

### Libraries Used:
- `numpy`
- `tensorflow`

### Output Explanation:
- The model predicts the next value in the Fibonacci sequence.
- The output is a single value, which represents the next Fibonacci number in the sequence based on the learned patterns from the training data.

---

## Project 2: Sentiment Prediction from Fairy Tale Text

This project uses LSTM to predict sentiment polarity from a classic fairy tale: **Hansel and Gretel**. Sentences are analyzed using TextBlob to determine their sentiment, which is then used as input for sequence modeling.

### Key Steps:
1. Load and preprocess the full text of *Hansel and Gretel*.
2. Split the story into individual sentences.
3. Perform sentiment analysis on each sentence using TextBlob.
4. Build sequences of sentiment scores for training. The sentiment polarity values are used as input features, and the model predicts future sentiment trends.
5. Train an LSTM model to predict future sentiment.

### Model Architecture:
- Input: shape `(time_steps, 1)`
- LSTM layer with 60 units and tanh activation
- Dense output layer
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam

### Libraries Used:
- `numpy`
- `re`
- `matplotlib` (imported but unused)
- `textblob`
- `tensorflow`

### Output and Plot Explanation:

#### Sentiment Score Plot:
The sentiment for each sentence in Hansel and Gretel is calculated using TextBlob, resulting in sentiment polarity values ranging from -1 (negative sentiment) to 1 (positive sentiment).
A plot can be created that visualizes the sentiment score of each sentence in the fairy tale. The x-axis represents the sentence index (from the story), and the y-axis represents the sentiment score. This allows us to observe how the sentiment changes throughout the story.


#### LSTM Model Prediction:
The model is trained on sequences of sentiment values and attempts to predict future sentiment based on the past sequence.
After training, the output is a prediction of the sentiment trend for the upcoming sentences in the story. The predicted sentiment values can be plotted to compare against the original sentiment values from the analysis. This allows us to see how well the model learned the trends in sentiment and whether it predicts future sentiment correctly.


#### Interpretation:

The red dashed line marks the boundary of the last sequence of sentences used for training the LSTM model, indicating the point at which the prediction begins. The green dashed line represents the predicted sentiment for the next part of the story, plotted as a horizontal line.

If the model performs well, the intersection point of the red and green lines should align closely with the point on the blue graph (actual sentiment values) that shares the same x-axis value. This indicates the model's prediction matches the true sentiment trend.

---


