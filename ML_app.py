import gradio as gr
import pickle
from collections import Counter

# Load the CountVectorizer
with open("count_vectorizer.pkl", "rb") as file:
    count_vectorizer = pickle.load(file)

# Load the models
models = {}
model_filenames = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Multinomial Naive Bayes": "multinomial_nb_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

for name, filename in model_filenames.items():
    with open(filename, "rb") as file:
        models[name] = pickle.load(file)

# Prediction function
def predict_news(input_text):
    # Vectorize the input text
    input_vector = count_vectorizer.transform([input_text]).toarray()
    predictions = {}

    # Predict with each model
    for name, model in models.items():
        pred = model.predict(input_vector)[0]
        predictions[name] = 'Real News' if pred == 1 else 'Fake News'

    # Perform majority voting
    vote_count = Counter(predictions.values())
    final_prediction = vote_count.most_common(1)[0][0]

    # Return detailed results and the majority-vote prediction
    result = f"Predictions:\n"
    for model_name, outcome in predictions.items():
        result += f"{model_name}: {outcome}\n"
    result += f"\nFinal Prediction (majority vote): {final_prediction}"
    return result

# Gradio interface setup
interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=4, placeholder="Enter news text here..."),
    outputs=gr.Textbox(label="Prediction"),
    title="Fake News Detection with Multi-ML models",
    description="Enter a news article or statement to classify it as real or fake based on predictions from multiple models."
)

# Launch the Gradio app
interface.launch()
