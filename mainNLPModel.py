

## COURSE RATER CHAT BOT TRAINED ON SOME MCGILL UNIVERSITY'S REVIEWS OF COMPUTER SCIENCE COURSES + RANDOMLY GENERATED REVIEWS ...
# MAIN MACHINE LEARNING MODEL CODE (TRAINING / TESTING / BOT'S INTERACTION WITH THE USER)

### NECESSARY IMPORTS:
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import warnings
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import lightgbm as lgb
import json
import random
import joblib
import torch
#from transformers import T5ForConditionalGeneration, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer
import openai
import base64
from gtts import gTTS
import os

warnings.filterwarnings('ignore')

# Ensure required NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

# Load the dataset
df = pd.read_csv("./combined_course_reviews.csv")
#st.write(f"Dataset loaded with shape: {df.shape}")

# Shuffle the data in the dataframe multiple times to ensure that the samples are randomly distributed
for _ in range(10):
    df_sampled = df.sample(frac=1, random_state=50).reset_index(drop=True)

### UTILIZATION OF 'MPS' --> Beneficial for MAC users:
# Utilize the Metal Performance Shaders (MPS) backend for GPU acceleration if available; otherwise, the system will default to using the CPU.
# MPS is a framework developed by Apple, specifically designed to accelerate GPU tasks on macOS and iOS devices. It is optimized for Apple Silicon chips (M1, M2) ...
# ... and is also compatible with certain AMD GPUs on macOS. By using MPS, you can achieve better performance, especially for demanding tasks like deep learning on Macs.
# Note that MPS is only available on Apple devices. If this model is run on a non-Mac system, it will default to the CPU, which may result in slower performance due to the lack of GPU acceleration.
# IMPORTANT: For the best performance, it's recommended to run this model on a Mac with 'MPS' support, OTHERWISE by default, it would run on the CPU with greater time consumption.

# IMP: SINCE I OWN A MAC, I WOULD BE USING THE 'MPS' FOR INCREASED PERFORMANCE BY THE GPU ... :)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#st.write(f"Model is running on: {device}")


# Text preprocessing function
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.analyzer = SentimentIntensityAnalyzer()

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def sentiment_score(self, text):
        return self.analyzer.polarity_scores(text)['compound']


# Initialize text preprocessor
text_preprocessor = TextPreprocessor()

# Preprocess the text data and add sentiment scores
df_sampled['Cleaned Review'] = df_sampled['Review'].apply(text_preprocessor.preprocess_text)
df_sampled['sentiment_score'] = df_sampled['Cleaned Review'].apply(text_preprocessor.sentiment_score)

# Use TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf_vectorizer.fit_transform(df_sampled['Cleaned Review']).toarray()

# Combine TF-IDF features with sentiment scores
X = np.hstack([X_tfidf, df_sampled[['sentiment_score']].values])


# Define the function to train and save models
def train_and_save_models(df_sampled, X, targets):
    models = {
        'CatBoost': CatBoostClassifier(random_state=50, verbose=0),
        'Neural Network': Pipeline([
            ('mlp', MLPClassifier(random_state=50, max_iter=1000))]),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=50),
        'Gradient Boosting': GradientBoostingClassifier(random_state=50),
        'K-Nearest Neighbors': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]),
        'LightGBM': lgb.LGBMClassifier(random_state=50, verbosity=-1),
        'Support Vector Classifier': Pipeline(
            [('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=50))]),
        'Random Forest': RandomForestClassifier(random_state=50),
    }

    param_grids = {
        'CatBoost': {
            'iterations': [50, 100],
            'learning_rate': [0.01, 0.1],
            'depth': [3, 4, 5],
            'l2_leaf_reg': [1, 3, 5]
        },
        'Neural Network': {
            'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'mlp__activation': ['relu', 'tanh'],
            'mlp__alpha': [0.0001, 0.001],
            'mlp__learning_rate_init': [0.001, 0.01]
        },
        'Logistic Regression': {
            'C': [0.01, 0.1, 0.2, 0.25, 0.3, 0.35, 0.5, 1, 5, 10, 15],
            'solver': ['lbfgs', 'liblinear', 'newton-cg'],
        },
        'K-Nearest Neighbors': {
            'knn__n_neighbors': [3, 5, 7, 10, 15, 20, 30, 35],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan', "hamming", "cosine", "minkowski"]
        },
        'Support Vector Classifier': {
            'svc__C': [0.01, 0.05, 1],
            'svc__kernel': ['linear', 'rbf']
        },
        'LightGBM': {
            'n_estimators': [30, 35, 40, 45],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [30, 50],
            'max_depth': [10, 20]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1],
            'max_depth': [3, 4]
        }
    }

    stacked_classifiers = {}

    for target in targets:
        st.write(f"\nTraining and evaluating models for {target}")

        X_train, X_test, y_train, y_test = train_test_split(X, df_sampled[target], test_size=0.20, random_state=50,
                                                            stratify=df_sampled[target])

        best_models = {}
        for model_name in models:
            st.write(f"\nTraining and evaluating {model_name} model for {target}")

            model = models[model_name]
            param_grid = param_grids[model_name]

            # LET'S PERFORM GRID SEARCH TO EXPLORE THE EXPANDED PARAMETER GRIDS EFFECTIVELY
            search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_models[model_name] = best_model

            # Evaluate the model
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Print the results
            st.write(f'Accuracy for {target}: {accuracy:.4f}')
            st.write(f'F1 Score for {target}: {f1:.4f}')
            st.write("-" * 30)

        # Create an ensemble model using StackingClassifier to combine the predictions from all models ...
        estimators = [(name, best_models[name]) for name in best_models]
        stack_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
        stack_clf.fit(X_train, y_train)
        y_pred_stack = stack_clf.predict(X_test)
        stack_accuracy = accuracy_score(y_test, y_pred_stack)

        st.write(f'\nStacking Classifier Accuracy for {target}: {stack_accuracy:.4f}')

        stacked_classifiers[target] = stack_clf

        # Save the stacked classifier for a particular target into a pkl file.
        joblib.dump(stack_clf, f'stacked_clf_{target}.pkl')

    return stacked_classifiers


# Function to load or train model --> for the training part: (if the model had not been previously trained ...)
def load_or_train_models():
    targets = ['Prof Rating', 'Course Rating']
    stacked_classifiers = {}

    # Safeguarding against redundant training by checking if the model has already been trained.
    try:
        for target in targets:
            stacked_classifiers[target] = joblib.load(f'stacked_clf_{target}.pkl')
            #st.write(f"Successfully Loaded saved model for {target}!")

    # Ensuring that the model undergoes training only if it has not been previously trained.
    except FileNotFoundError:
        st.write("Saved models not found, training models...")
        stacked_classifiers = train_and_save_models(df_sampled, X, targets)

    return stacked_classifiers


stacked_classifiers = load_or_train_models()

# Load the JSON file to get the keyword(s) corresponding to each rating.
with open('extended_ratings_descriptions.json', 'r') as file:
    ratings_descriptions = json.load(file)


# Function to generate extended review paragraph
def generate_extended_review_paragraph(prof_rating, course_rating):
    if (prof_rating == -1 and course_rating == -1):
        return None
    elif prof_rating == -1:
        prof_desc = "\nThe Professor's review could not be found (is NOT-AVAILABLE) in the user-entered comment."
        st.write(prof_desc)

        course_desc = random.choice(ratings_descriptions[str(course_rating)])
        st.write("However, in consideration of the 'AI-GENERATED' Course Rating provided above, we present a 'comprehensive' review that encapsulates the essence of the course as experienced by the students, reflecting on key elements such as course content, instructional quality, and the overall learning environment.")
           
        prompt_to_feed_in_API = f"The Course's Review in the user-mentioned comment is {course_desc}."

    elif course_rating == -1:
        course_desc = "\nThe course's review could not be found (is NOT-AVAILABLE) in the user-entered comment ..."
        st.write(course_desc)

        prof_desc = random.choice(ratings_descriptions[str(prof_rating)])
        st.write(
            "However, considering the above 'AI-GENERATED' Professor Rating, here is a 'thoughtfully crafted' summary that captures the essence of students' experiences and interactions with the professor:")

        prompt_to_feed_in_API = f"The Professor's Review in the user-mentioned comment is {prof_desc}."
    else:
        prof_desc = random.choice(ratings_descriptions[str(prof_rating)])
        course_desc = random.choice(ratings_descriptions[str(course_rating)])
        st.write(
            "\nTaking into account the above 'AI-GENERATED' ratings about the Professor and Course, presented below is a very 'concise' summary of the students' experiences with both the Course and the Professor:")
        prompt_to_feed_in_API = (
            f"The professor's review in the user-mentioned comment is described as {prof_desc}. The course's review in the user-mentioned comment is described as {course_desc}.")

        ### ENHANCED PARAGRAPH GENERATION ACCORDING TO THE AI PREDICTED REVIEWS ...
    ## TRYING 3 TECHNIQUES WITH PARAGRAP GENERATION:

    # This section of code generates 5 different outputs using the pre trained T5 model, each offering a unique variation of the response.
    # We then iterate through these outputs, decoding and comparing their lengths to identify the one with the highest character count.
    # GOAL -> To select and print the most detailed and comprehensive output, ensuring that the final result captures the fullest elaboration possible.

    ###  (1) Using T5 Tokenizer Technique: --> Print the longest review out of the 5 generated reviews.
    # Load the T5 tokenizer and model
    # t5_tokenizer = T5Tokenizer.from_pretrained('t5-base', legacy = False)
    # t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

    # # Tokenize input text for T5 model
    # input_ids = t5_tokenizer.encode(prompt_to_feed_in_API , return_tensors='pt').to(device)

    # # Use torch.no_grad() to avoid unnecessary gradient computation
    # with torch.no_grad():
    #     # Generate output with optimized settings

    #     outputs = t5_model.generate(
    #     input_ids,
    #     max_length=500,           # Increase length if needed for more detailed output
    #     num_return_sequences=5,   # Generate 5 outputs
    #     no_repeat_ngram_size=2,   # Prevent redundant phrases
    #     temperature=0.70,          # Maintain a balance between creativity and coherence
    #     top_k=20,                 # Allow some diversity in token selection
    #     top_p=0.9,                # Use nucleus sampling for nuanced output
    #     do_sample=True            # Enable sampling for varied results
    # )

    # longest_review = ""
    # max_length = 0
    # for i, output in enumerate(outputs):
    #     result = (t5_tokenizer.decode(output, skip_special_tokens=True))
    #     print(result)
    #     translator = Translator()
    #     original_texts = translator.translate(result, src='auto', dest='en')
    #     decoded_output = original_texts.text
    #     print(decoded_output)
    #     output_length = len(decoded_output)
    #     if output_length >= max_length:
    #         longest_review = decoded_output
    #         max_length = output_length

    # return longest_review

    ####  (2) Using the GPT2 (gpt2 tokenizer + distilgpt2 model) technique:
    # gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', legacy = False)
    # # Set pad_token to eos_token
    # gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    # gpt2_model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(device)

    # input_text = prompt_to_feed_in_API
    # # Tokenize the input review on CPU
    # inputs = gpt2_tokenizer.encode(input_text, return_tensors='pt', padding = True)
    # # Move to MPS -> (if available)
    # inputs = inputs.to(device)
    # attention_mask = (inputs != gpt2_tokenizer.pad_token_id).long().to('mps')

    # with torch.no_grad():
    #     outputs = gpt2_model.generate(
    #     inputs,
    #     attention_mask = attention_mask,
    #     max_length=30,       # Max length of the output text
    #     num_return_sequences=1, # Number of generated sequences
    #     no_repeat_ngram_size=3, # Prevents repetition of n-grams
    #     temperature=0.5,        # Controls the randomness (lower = more focused, higher = more creative)
    #     top_p=0.80,             # Controls nucleus sampling
    #     top_k=30               # Limits the number of highest probability vocabulary tokens
    # )

    # # Decode the generated text
    # generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(generated_text)
    # return generated_text

    # (3) Use OPENAI API for elaborated generation of the prompt_to_feed_in_API.
    
    from config import Shahmeer_OpenAI_API_Key   # Importing my own API key from config.py --> config.py has been for safety purposes.
    openai.api_key = Shahmeer_OpenAI_API_Key  # Setting up my own API Key for OpenAI.

    # Optionally, list all available models, using the following code:
    # print("The available models are:\n")
    # models = openai.Model.list()
    # for model in models['data']:
    #     print(model['id'])
    # print("end ...")

    ## --> RESULT:
    # The available models are: dall-e-3, gpt-4-1106-preview, dall-e-2, tts-1-hd-1106, tts-1-hd,
    # text-embedding-3-large, babbage-002, gpt-4-turbo-preview, gpt-4o-mini, gpt-4-0125-preview,
    # gpt-4o-mini-2024-07-18, text-embedding-3-small, tts-1, gpt-3.5-turbo, whisper-1,
    # text-embedding-ada-002, gpt-3.5-turbo-16k, davinci-002, gpt-4-turbo-2024-04-09, tts-1-1106,
    # gpt-3.5-turbo-0125, gpt-4-turbo, gpt-3.5-turbo-1106, gpt-3.5-turbo-instruct-0914,
    # gpt-3.5-turbo-instruct, gpt-4o, gpt-4-0613, gpt-4o-2024-05-13, gpt-4, gpt-4o-2024-08-06
    # end ...

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are a knowledgeable and articulate assistant, skilled at providing clear, professional, and detailed explanations."},
            {"role": "user",
             "content": f"Please enhance and elaborate on the following text. Provide a well-structured and concise paragraph that clearly articulates the course or professor rating, ensuring the explanation is both comprehensive and polished for the user: {prompt_to_feed_in_API}"}
        ]
    )

    elaborated_text = response['choices'][0]['message']['content']
    return elaborated_text


# Function to predict ratings
def predict_ratings(review):
    review_vectorized = tfidf_vectorizer.transform([review]).toarray()
    review_combined = np.hstack([review_vectorized, np.array([[text_preprocessor.sentiment_score(review)]])])

    prof_rating = int(stacked_classifiers['Prof Rating'].predict(review_combined)[0])
    course_rating = int(stacked_classifiers['Course Rating'].predict(review_combined)[0])

    return prof_rating, course_rating



### DEPLOYING A WEBSITE USING STREAMLIT ...

#background_image_path = "./background_images/openart-image_ymjj0xoi_1723478456964_raw.jpg"
background_image_path = "./background_images/chatbot's_background_image.jpg"
#background_image_path = "./background_images/openart-image_N-7ZkR4H_1723482040606_raw.jpg"
#background_image_path = "./background_images/openart-image_BJ0_Igjs_1723483659365_raw.jpg"
# Encode the background image
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# Load your background image and encode it
encoded_image = get_base64_encoded_image(background_image_path)
# Set up custom CSS with the base64 image
st.markdown(

    f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpeg;base64,{encoded_image});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#st.title("Dr. RateMate")
st.markdown("<h1 style='font-size: 100px; color: #4B0082;'><i>Dr. RateMate</i></h1>", unsafe_allow_html=True)
st.markdown("""
    <p style='color: #000000; font-size: 24px; font-family: Arial, sans-serif; font-weight: bold;'>
    <strong>Are you overwhelmed by chaotic and hard-to-understand course and professor reviews?</strong>
    </p>
    <p style='color: #000000; font-size: 24px; font-family: Arial, sans-serif; font-weight: bold;'>
    <strong>Our Chat-Bot is here to help!</strong>
    </p>
    <p style='color: #000000; font-size: 24px; font-family: Arial, sans-serif; font-weight: bold;'>
    <strong>Simply provide a review, and watch as Dr. RateMate transforms your feedback into clear, actionable ratings for both the professor and the course.</strong>
    </p>
    <p style='color: #000000; font-size: 24px; font-family: Arial, sans-serif; font-weight: bold;'>
    <strong>It doesn’t just crunch numbers; it breaks down complex reviews into easily digestible insights, making your decision-making process smoother and more informed.</strong>
    </p>
    <p style='color: #000000; font-size: 24px; font-family: Arial, sans-serif; font-weight: bold;'>
    <strong>Say goodbye to confusion and hello to a clearer academic path!</strong>
    </p>
    """, unsafe_allow_html=True)

user_review = st.text_area("Go on, we know you've got a review bubbling up inside! Whether it’s a course that made you question everything or a professor who made you wish you’d paid more attention in class, let it all out. Your words could change the future… or at least help someone pick their next class. Ready to spill the tea? If not, just type 'quit' and save the drama for another day.")

if user_review.lower() != 'quit':
    if st.button("Dive into the Feedback"):
        prof_rating, course_rating = predict_ratings(user_review)

        st.write(f"The Professor's rating for the user-entered comment is: {prof_rating}")
        st.write(f"The Course's rating for the user-entered comment is: {course_rating}")

        # Generate a creative paragraph about the review for the user
        generated_review = generate_extended_review_paragraph(prof_rating, course_rating)

        if generated_review is None:
            st.write("Oops! It seems the professor's review has gone missing from the comment you've entered.")
            st.write("And it looks like the course's review has also decided to take a break and is nowhere to be found in your input.")
            st.write(
                "To help us out, could you please provide a more detailed and insightful comment? We'd love to hear your thoughts—both on the professor and the course. Don't leave us hanging!")

        else:
            st.write(generated_review)
            
            # Assuming generated_review is defined
            tts = gTTS(text=generated_review, lang='en')
            tts.save("Bot-Review.mp3")

            # Check if the file exists
            if os.path.exists("Bot-Review.mp3"):
                print("Bot-Review.mp3 has been created successfully.")
                
                # Read the audio file and play it
                with open("./Bot-Review.mp3", "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")

            else:
                print("Failed to create Bot-Review.mp3.")
