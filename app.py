from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

with open('recipe_recommendation_model.pkl', 'rb') as model_file:
    tfidf_vectorizer, tfidf_matrix, df = pickle.load(model_file)

def recommend_similar_by_ingredients(recipe_index):
    """
    Recommend similar recipes based on ingredients of the recipe at 'recipe_index'.
    """
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[recipe_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:9]  
    recipe_indices = [i[0] for i in sim_scores]
    recommended_recipes = df[['recipe_title', 'ingredients', 'instructions', 'tags', 'category', 'cuisine', 'course', 'diet']].iloc[recipe_indices]

    recommended_recipes['ingredients'] = recommended_recipes['ingredients'].str.replace('|', ', ')
    recommended_recipes['tags'] = recommended_recipes['tags'].str.replace('|', ', ')

    return recommended_recipes


@app.route('/')
def home():
    cuisines = df['cuisine'].dropna().unique()
    courses = df['course'].dropna().unique()
    diets = df['diet'].dropna().unique()

    return render_template('index.html', cuisines=cuisines, courses=courses, diets=diets)

@app.route('/recommend', methods=['POST'])
def recommend():
    cuisine = request.form.get('cuisine', '').strip()
    course = request.form.get('course', '').strip()
    diet = request.form.get('diet', '').strip()

    filtered_df = df
    if cuisine:
        filtered_df = filtered_df[filtered_df['cuisine'].str.lower() == cuisine.lower()]
    if course:
        filtered_df = filtered_df[filtered_df['course'].str.lower() == course.lower()]
    if diet:
        filtered_df = filtered_df[filtered_df['diet'].str.lower() == diet.lower()]

    if filtered_df.empty:
        return render_template('index.html', error="No recipes found with the selected filters.", cuisines=cuisine, courses=course, diets=diet)
    recipe_index = filtered_df.index[0]
    recommended_recipes = recommend_similar_by_ingredients(recipe_index)
    
    return render_template('recommendations.html', recipes=recommended_recipes)

if __name__ == '__main__':
    app.run(debug=True)
