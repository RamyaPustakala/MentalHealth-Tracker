import streamlit as st
import googlemaps
import folium
from streamlit_folium import folium_static
from streamlit_geolocation import streamlit_geolocation

# Initialize Google Maps client
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
    map = folium.Map(location=map_center, zoom_start=15)

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

def main():
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

if __name__ == "__main__":
    main()
