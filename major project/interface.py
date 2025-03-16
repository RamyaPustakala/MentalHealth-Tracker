import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
import webbrowser
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import googlemaps
import folium
from io import BytesIO
from streamlit_geolocation import streamlit_geolocation
from streamlit_folium import folium_static


database = 'project_db.sqlite'  # SQLite database file

gmaps = googlemaps.Client(key='AIzaSyAQqwrjtX7HGRur4y8E6nogj3slNllH7KI')  # Replace 'YOUR_API_KEY' with your actual Google Maps API key

# Function to fetch nearby psychiatrists using Google Places API
def fetch_nearby_psychiatrists(location, radius):
    places_result = gmaps.places(
        query='psychiatrist',
        location=location,
        radius=radius
    )
    return places_result['results']

# Function to display nearby psychiatrists on Folium map
def display_psychiatrists_map(psychiatrists, user_location):
    map_center = (user_location['latitude'], user_location['longitude'])
    map = folium.Map(location=map_center, zoom_start=12)

    # Add marker for user's location
    folium.Marker(
        location=map_center,
        popup="Your Location",
        icon=folium.Icon(color='red')
    ).add_to(map)

    # Add markers for nearby psychiatrists
    for psychiatrist in psychiatrists:
        name = psychiatrist['name']
        address = psychiatrist['formatted_address']
        latitude = psychiatrist['geometry']['location']['lat']
        longitude = psychiatrist['geometry']['location']['lng']
        folium.Marker([latitude, longitude], popup=f"{name}\n{address}").add_to(map)

    return map

def location():
    st.header("Find Nearby Psychiatrists")
    
    # Get the user's current location
    user_location = streamlit_geolocation()

    if user_location['latitude'] is not None:
        user_latitude = user_location['latitude']
        user_longitude = user_location['longitude']
        
        # Fetch nearby psychiatrists
        psychiatrists = fetch_nearby_psychiatrists((user_latitude, user_longitude), radius=10000)  # 10km radius
        
        # Display names and addresses of nearby psychiatrists
        st.write("## Nearby Psychiatrists")
        for psychiatrist in psychiatrists:
            name = psychiatrist['name']
            address = psychiatrist['formatted_address']
            st.write(f"**Name:** {name}")
            st.write(f"**Address:** {address}")
            st.write("---")

        # Display the map with nearby psychiatrists and user's location
        st.write("## Map of Nearby Psychiatrists")
        map = display_psychiatrists_map(psychiatrists, user_location)
        folium_static(map)
    else:
        st.warning("Please grant location permission to find nearby psychiatrists.")


def ask_questions():
    # Define the questions
    questions = [
        "How often did you experience general feelings of anxiety (nervousness, worry, tension) in the past week",
        "How often did you experience physical symptoms of anxiety (sweating, trembling, heart racing) in the past week",
        "How often did you feel down, sad, or hopeless in the past week?",
        "How often did you lose interest in activities you used to enjoy?",
        "How often did you feel easily annoyed or frustrated in the past week?",
        "How often did you worry about panicking or losing control in the past week?",
        "How often did you find it difficult to get going or feel sluggish?",
        "How often did you feel like overreacting to situations?",
        "How often did you have difficulty relaxing?",
        "How often did you feel scared or frightened without a good reason?",
        "How often did you find it difficult to swallow or breathe?",
    ]

    # Ask the questions and get the answers
    options = [
        "Did not apply to me at all",
        "Applied to me to some degree, or some of the time",
        "Applied to me to a considerable degree, or a good part of the time",
        "Applied to me very much, or most of the time"
    ]

    answers = []
    print(len(questions))
    for question in questions:
        answer = st.radio(question, options, index=None)
        if answer is not None:
            answers.append(options.index(answer) + 1)  # Convert selected option to its index (1-indexed)
        else:
            answers.append(None)

    # Check the length of the answers list
    print("Length of answers:", len(answers))

    # Calculate the average of the answers
    mapping = {
        1: 1, 12: 1, 14: 1, 18: 1, 32: 1, 33: 1, 35: 1,
        2: 2, 7: 2, 15: 2, 19: 2, 23: 2, 25: 2, 41: 2,
        3: 3, 10: 3, 13: 3, 16: 3, 21: 3, 26: 3, 37: 3, 38: 3,
        24: 4, 31: 4,
        11: 5, 27: 5, 39: 5, 9: 5,
        28: 6, 40: 6,
        5: 7, 42: 7,
        6: 8, 34: 8, 17: 8,
        8: 9, 22: 9, 29: 9,
        20: 10, 36: 10, 30: 10,
        23: 11, 4: 11
    }
    
    values = []
    dummy_values = [mapping.get(i, 0) for i in range(1, 43)]
    print(dummy_values)
    for i in dummy_values:
        k = i - 1
        values.append(answers[k])
    print(values)

    return values


def predict_with_model(X_test, condition):
    model = load_model(f"{condition}_cnn_model.keras")
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)  # Get class with highest probability
    return predicted_classes

def show_results(values):
    indices = {
    'Depression': [2, 4, 9, 12, 15, 16, 20, 23, 25, 30, 33, 36, 37, 41], 
    'Anxiety': [1, 3, 6, 8, 14, 18, 19, 22, 24, 27, 29, 35, 39, 40], 
    'Stress': [0, 5, 7, 10, 11, 13, 17, 21, 26, 28, 31, 32, 34, 38]
    }


    Depression_test = [values[i] for i in indices['Depression']]
    Stress_test = [values[i] for i in indices['Stress']]
    Anxiety_test = [values[i] for i in indices['Anxiety']]

    classes = ["Extremely Severe","Severe","Moderate","Mild","Normal"]


    X_depression_test = np.array(Depression_test).reshape(1, len(Depression_test), 1)
    p_d = predict_with_model(X_depression_test, 'Depression')
    print(p_d)
    X_stress_test = np.array(Stress_test).reshape(1, len(Stress_test), 1)
    p_s = predict_with_model(X_stress_test, 'Stress')
    print(p_s)
    X_anxiety_test = np.array(Anxiety_test).reshape(1, len(Anxiety_test), 1)
    p_a = predict_with_model(X_anxiety_test, 'Anxiety')
    print(p_a)

    depression_sevirity = [classes[i] for i in p_d]
    stress_sevirity = [classes[i] for i in p_s]
    anxiety_sevirity = [classes[i] for i in p_a]

    aavg = p_d + p_s + p_a
    print(aavg)

    #print(question_values)
    average = sum(aavg) / 3
    print(average)

    st.markdown('##')
    # Display the average on the page
    st.write(f"Depression serveritiy is:  {depression_sevirity[0]}" )
    st.write(f"Stress serveritiy is:  {stress_sevirity[0]}" )
    st.write(f"Anxiety serveritiy is:  {anxiety_sevirity[0]}" )
    st.write(f"Your average mental health score today is {average}")

    # Add the gauge chart to show where the average score lies on a scale of 0 to 10
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=average,
        mode="gauge+number",
        title={'text': "Mental Health Score"},
        gauge={
            'axis': {'range': [0, 4]},
            'steps': [
                {'range': [3, 4], 'color': "green"},
                {'range': [2, 3], 'color': "orange"},
                {'range': [0, 1], 'color': "red"},
                ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': average}}))

    st.plotly_chart(fig, use_container_width=True, height=50)

    if average <= 2:
        location()

    # Get the current date and time
    now = datetime.now()

    # Format the date as a string in the format "YYYY-MM-DD"
    date_string = now.strftime('%Y-%m-%d')

    # Add the entry to the database
    if st.button('Submit to your daily MindLens tracker'):
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS mental_health (date TEXT, Depression INT, Anxiety INT, Stress INT, average FLOAT)")
        cursor.execute("INSERT INTO mental_health (date, Depression, Anxiety, Stress, average) VALUES (?, ?, ?, ?, ?)", 
                       (date_string, int(p_d[0]), int(p_a[0]), int(p_s[0]), average))
        conn.commit()
        conn.close()
        st.write("Your mental health check-in has been submitted!")
def read_data():
    conn = sqlite3.connect(database)
    df = pd.read_sql_query("SELECT * FROM mental_health ORDER BY date DESC", conn)
    conn.close()
    return df

def get_average_scores():
    conn = sqlite3.connect(database)
    df2 = pd.read_sql_query(
        "SELECT AVG(Depression) as avg_Depression, AVG(Anxiety) as avg_Anxiety, AVG(Stress) as avg_Stress FROM mental_health", conn)
    conn.close()
    return df2.values[0]

def get_average_scores_dataframe(average_scores):
    df3 = pd.DataFrame({
        "category": ["Depression", "Anxiety", "Stress"],
        "average": average_scores
    })
    return df3

def show_dataframe():
    st.write("## Mental Health Data")
    df = read_data()
    st.dataframe(df)

def delete_data():
    if st.button("Delete all data"):
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS mental_health")
        conn.commit()
        conn.close()
        st.write("All data has been deleted from the database.")

def show_visualization(conn):
    with conn:
        # Generate the visualizations
        df = read_data()
        fig1 = px.line(df, x="date", y="average", line_shape="spline", color_discrete_sequence=["red"])
        fig1.update_layout(xaxis_tickformat='%Y-%m-%d', title="Average Mental Health Score Over Time")

        fig2 = px.line(df, x="date", y=["Depression", "Anxiety", "Stress"], line_shape="spline")
        fig2.update_layout(xaxis_tickformat='%Y-%m-%d', title="Mental Health Scores Over Time")

        # Get and plot the average scores
        average_scores = get_average_scores()
        df3 = get_average_scores_dataframe(average_scores)
        fig3 = px.bar_polar(df3, r="average", theta="category", template="plotly_dark")
        fig3.update_traces(opacity=0.7)
        fig3.update_layout(title="Average Mental Health Scores by Category")

    # Show the visualizations on the page
    show_dataframe()
    delete_data()
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)

def show_guidance(conn):
    st.write("## Mental Health Guidances")
    df = read_data()
    now = datetime.now()

# Format the date as a string in the format "YYYY-MM-DD"
    date_string = now.strftime('%Y-%m-%d')

    Dp = df[df['date'] == date_string]['Depression'].values[0]
    print(Dp)

    ss = df[df['date'] == date_string]['Stress'].values[0]
    print(ss)

    an = df[df['date'] == date_string]['Anxiety'].values[0]
    print(an)

    if Dp < 1:
        st.title("Yoga poses to control Depression")

        tree_image = Image.open("tree.jpg")
        dog_image = Image.open("dog.png")

    # Display images
        st.image(tree_image, caption='Tree Image', use_column_width=True)
        st.image(dog_image, caption='Dog Image', use_column_width=True)

    if ss < 1:
        st.title("Yoga poses to control Stress")

        tree_image = Image.open("cobra.jpg")
        dog_image = Image.open("warrior.png")

    # Display images
        st.image(tree_image, caption='Cobra Pose', use_column_width=True)
        st.image(dog_image, caption='Warrior Pose', use_column_width=True)

    if an < 2:
        st.title("Yoga poses to control Anxiety")

        tree_image = Image.open("chair.jpg")
        dog_image = Image.open("shoulderstand.jpg")

    # Display images
        st.image(tree_image, caption='Chair Pose', use_column_width=True)
        st.image(dog_image, caption='Shoulderstand Pose', use_column_width=True)

    if Dp >= 1 and ss >= 1 and an >= 2:
        st.write("You are doing well, continue practicing Yoga and beat your best.")
    

def main():
    st.set_page_config(
        page_title="Mental Health Tracker",
        page_icon=":brain:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Mental Health Tracker")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Go to")
    page = st.sidebar.radio("Navigate to", ["Check-In", "Data", "Guidance"])

    if page == "Check-In":
        values = ask_questions()
        if all(value is not None for value in values):  # Check if all values are not None
            show_results(values)
        else:
            st.write("Please answer all questions to proceed.")
    elif page == "Data":
        show_visualization(sqlite3.connect(database))
    elif page == "Guidance":
        show_guidance(sqlite3.connect(database))
if __name__ == "__main__":
    main()
