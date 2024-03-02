from fastapi import FastAPI, Form
from pydantic import BaseModel, Field
from typing import Optional ,List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random

master = {}

class Db(BaseModel):
    Country: str = Form(..., description="Country label")
    Governorate: str = Form(None, description="Governorate label")
    Survey: List[str] = Form([], description="Survey label")
    Total_Price: float = Form(None, description="Total Price label")
app = FastAPI()

# Sample Data
selected_columns = ['Title', 'Tag', 'Review', 'Comment', 'Address', 'Country', 'Price', 'Rating', 'tags', 'Governorate']
df = pd.read_csv(r"final_data.csv")  
df = df[selected_columns].dropna()

df['Tag'] = df['Tag'].astype(str)
df['Review'] = df['Review'].astype(str)
df['Comment'] = df['Comment'].astype(str)

# Feature Engineering
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Tag'] + ' ' + df['Review'] + ' ' + df['Comment'])

# Model Selection
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Split the dataset
train, test = train_test_split(df, test_size=0.2, random_state=42)

def get_recommendations(country, governorate, survey_responses, df):
    # Filter places that match the user's country and governorate
    filtered_df = df[(df['Country'] == country) & (df['Governorate'] == governorate)]

    # Check if there is data available
    if filtered_df.empty:
        print("No data found for the specified country and governorate.")
        return pd.DataFrame(columns=['Title', 'Price', 'tags', 'Governorate'])

    # Ensure "Restaurants" and "Hotels" are always present in user profile
    user_profile = f"{country} {governorate} {' '.join(survey_responses)}"

    # Transform the user profile using the same TF-IDF vectorizer
    user_profile_vectorized = tfidf_vectorizer.transform([user_profile])

    # Transform places' descriptions (assuming 'tags' contains relevant information)
    places_vectorized = tfidf_vectorizer.transform(filtered_df['tags'])

    # Calculate cosine similarity between user profile and places
    sim_scores = linear_kernel(user_profile_vectorized, places_vectorized).flatten()

    # Create a DataFrame to store recommendations
    recommendations_df = pd.DataFrame(columns=['Title', 'Price', 'tags', 'Governorate'])

    # Check if there are places to recommend
    if not any(response_indices for response in survey_responses for response_indices in [i for i, tag in enumerate(filtered_df['tags']) if response.lower() in tag.lower()]):
        print("No suitable places found for the given survey responses.")
        return recommendations_df

    # Iterate through each survey response and select a recommendation
    for response in survey_responses:
        # Get indices of places containing the current survey response
        response_indices = [i for i, tag in enumerate(filtered_df['tags']) if response.lower() in tag.lower()]

        # If there are places with the current tag, select a random one
        if response_indices:
            random_index = random.choice(response_indices)
            recommendation = filtered_df.iloc[[random_index]][['Title', 'Price', 'tags', 'Governorate']]
            recommendations_df = pd.concat([recommendations_df, recommendation])

    # Add at least 3 random restaurant recommendations for each day
    for _ in range(1):
        restaurant_recommendation = filtered_df[filtered_df['tags'].str.lower().str.contains('restaurant')].sample(1)[['Title', 'Price', 'tags', 'Governorate']]
        recommendations_df = pd.concat([recommendations_df, restaurant_recommendation])

    # Add a hotel recommendation
    hotel_recommendation = filtered_df[filtered_df['tags'].str.lower().str.contains('hotel')].sample(1)[['Title', 'Price', 'tags', 'Governorate']]
    recommendations_df = pd.concat([recommendations_df, hotel_recommendation])

    return recommendations_df


@app.get("/master")
def get_db(limit: int = 2):
    db_list = list(master.values())
    return db_list[:limit]

@app.post("/master")
def create_db(data: Db):
    Country = data.Country
    Governorate = data.Governorate
    Survey = data.Survey

    # Check if the country and governorate exist in the dataset
    if (Country not in df['Country'].values) or (Governorate not in df['Governorate'].values):
        return {"message": f"Country or Governorate not found in the dataset."}

    # Get recommendations based on user input
    user_survey_responses = [] if Survey is None else (Survey if isinstance(Survey, list) else Survey.split(','))
    recommendations = get_recommendations(Country, Governorate, user_survey_responses, df)

    # Store the recommendations in the database
    master[Country] = recommendations.to_dict(orient='records')

    return {"message": f"Successfully created Trip with recommendations for {Country}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
