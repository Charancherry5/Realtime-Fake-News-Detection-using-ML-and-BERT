import pickle

# Load CountVectorizer
with open('count_vectorizer.pkl', 'rb') as file:
    count_vectorizer = pickle.load(file)

# Load each model

with open('logistic_regression_model.pkl', 'rb') as file:
    logistic_regression_model = pickle.load(file)

with open('random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

with open('multinomial_nb_model.pkl', 'rb') as file:
    multinomial_nb_model = pickle.load(file)

with open('xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# Dictionary to hold loaded models
loaded_models = {
    "Logistic Regression": logistic_regression_model,
    "Random Forest": random_forest_model,
    "Multinomial Naive Bayes": multinomial_nb_model,
    "XGBoost": xgb_model
}

# Function to make predictions with loaded models
def predict_news(loaded_models, count_vectorizer, input_text):
    # Transform the input text using the CountVectorizer
    input_vector = count_vectorizer.transform([input_text]).toarray()
    
    predictions = {}
    
    # Predict using each loaded model
    for name, model in loaded_models.items():
        pred = model.predict(input_vector)[0]
        predictions[name] = 'Real News' if pred == 1 else 'Fake News'
    
    return predictions

# Example input text
input_text = '''

Ever get the feeling your life circles the roundabout rather than heads in a straight line toward the intended destination? [Hillary Clinton remains the big woman on campus in leafy, liberal Wellesley, Massachusetts. Everywhere else votes her most likely to don her inauguration dress for the remainder of her days the way Miss Havisham forever wore that wedding dress.  Speaking of Great Expectations, Hillary Rodham overflowed with them 48 years ago when she first addressed a Wellesley graduating class. The president of the college informed those gathered in 1969 that the students needed â€œno debate so far as I could ascertain as to who their spokesman was to beâ€ (kind of the like the Democratic primaries in 2016 minus the   terms unknown then even at a Seven Sisters school). â€œI am very glad that Miss Adams made it clear that what I am speaking for today is all of us â€”  the 400 of us,â€ Miss Rodham told her classmates. After appointing herself Edger Bergen to the Charlie McCarthys and Mortimer Snerds in attendance, the    bespectacled in granny glasses (awarding her matronly wisdom â€”  or at least John Lennon wisdom) took issue with the previous speaker. Despite becoming the first   to win election to a seat in the U. S. Senate since Reconstruction, Edward Brooke came in for criticism for calling for â€œempathyâ€ for the goals of protestors as he criticized tactics. Though Clinton in her senior thesis on Saul Alinsky lamented â€œBlack Power demagoguesâ€ and â€œelitist arrogance and repressive intoleranceâ€ within the New Left, similar words coming out of a Republican necessitated a brief rebuttal. â€œTrust,â€ Rodham ironically observed in 1969, â€œthis is one word that when I asked the class at our rehearsal what it was they wanted me to say for them, everyone came up to me and said â€˜Talk about trust, talk about the lack of trust both for us and the way we feel about others. Talk about the trust bust.â€™ What can you say about it? What can you say about a feeling that permeates a generation and that perhaps is not even understood by those who are distrusted?â€ The â€œtrust bustâ€ certainly busted Clintonâ€™s 2016 plans. She certainly did not even understand that people distrusted her. After Whitewater, Travelgate, the vast   conspiracy, Benghazi, and the missing emails, Clinton found herself the distrusted voice on Friday. There was a load of compromising on the road to the broadening of her political horizons. And distrust from the American people â€”  Trump edged her 48 percent to 38 percent on the question immediately prior to Novemberâ€™s election â€”  stood as a major reason for the closing of those horizons. Clinton described her vanquisher and his supporters as embracing a â€œlie,â€ a â€œcon,â€ â€œalternative facts,â€ and â€œa   assault on truth and reason. â€ She failed to explain why the American people chose his lies over her truth. â€œAs the history majors among you here today know all too well, when people in power invent their own facts and attack those who question them, it can mark the beginning of the end of a free society,â€ she offered. â€œThat is not hyperbole. â€ Like so many people to emerge from the 1960s, Hillary Clinton embarked upon a long, strange trip. From high school Goldwater Girl and Wellesley College Republican president to Democratic politician, Clinton drank in the times and the place that gave her a degree. More significantly, she went from idealist to cynic, as a comparison of her two Wellesley commencement addresses show. Way back when, she lamented that â€œfor too long our leaders have viewed politics as the art of the possible, and the challenge now is to practice politics as the art of making what appears to be impossible possible. â€ Now, as the big woman on campus but the odd woman out of the White House, she wonders how her current station is even possible. â€œWhy arenâ€™t I 50 points ahead?â€ she asked in September. In May she asks why she isnâ€™t president. The woman famously dubbed a â€œcongenital liarâ€ by Bill Safire concludes that lies did her in â€”  theirs, mind you, not hers. Getting stood up on Election Day, like finding yourself the jilted bride on your wedding day, inspires dangerous delusions.

'''

# Get predictions
predictions = predict_news(loaded_models, count_vectorizer, input_text)
print('--------------------------------------------------')
# Display predictions
for model_name, prediction in predictions.items():
    print(f"{model_name}: {prediction}")
print('--------------------------------------------------')

