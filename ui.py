import sys
sys.path.append('/home/ch3370/ds/env/lib/python3.8/site-packages')
import numpy as np
import pandas as pd
import pickle
from surprise.dump import load
import folium
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import math

model = load('model/trained_model_svd.pkl')[1]
df_restaurant_exploded = pd.read_csv('processed_data/df_restaurant_exploded.csv')
with open('processed_data/categories_to_show.pkl','rb') as file:
    categories_to_show = pickle.load(file)
with open('processed_data/all_users.pkl','rb') as file:
    all_users = pickle.load(file)
with open('processed_data/all_users_dic.pkl','rb') as file:
    all_users_dic = pickle.load(file)   
with open('processed_data/all_restaurants_dic.pkl','rb') as file:
    all_restaurants_dic = pickle.load(file)  


def deg2rad(deg):
    return deg * math.pi/180
def get_distance(lat1,lon1,lat2,lon2):
    """
    caculate the distance in km
    """
    R = 6371
    dLat = deg2rad(lat1-lat2)
    dLon = deg2rad(lon1-lon2)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a),math.sqrt(1-a))
    d = R * c
    return d

from geopy.geocoders import OpenCage
geolocator = OpenCage(api_key="afb8f0336d004f678a72bce02e5a5e19")
def recommend(location, category, num_recomendations = 10, distance = 5, uid=None, ):
    print(location)
    recommendations = {}
    if location is not None:
        try:
            location = geolocator.geocode(location)
            user_lat, user_lon = location.latitude, location.longitude
        except:
            location,user_lat, user_lon = None,None,None
            print('Invalid address')
    category_filtered = df_restaurant_exploded[df_restaurant_exploded['categories'] == category]

    for _, row in category_filtered.iterrows():
        bid = all_restaurants_dic[row['business_id']]
        lat = row['latitude']
        lon = row['longitude']
        stars = row['stars']
        name = row['name']
        if location is not None:
            dist = get_distance(lat,lon,user_lat,user_lon)     
            if dist < distance:
                recommendations.update( {bid: (model.predict(all_users_dic.get(uid),bid).est, lat,lon,name,stars) })
        else:
            recommendations.update( {bid: (model.predict(all_users_dic.get(uid),bid).est, lat,lon,name,stars) })
    if uid in all_users_dic:
        sorted_rec = dict(sorted(recommendations.items(),key=lambda x:x[0],reverse=True)[:num_recomendations])
    else: 
        sorted_rec = dict(sorted(recommendations.items(),key=lambda x:x[-1],reverse=True)[:num_recomendations])
    return sorted_rec  


def update_recommendations(change):
    location = location_widget.value
    uid = uid_widget.value
    category = category_widget.value
    distance = distance_widget.value
    num_recommendations = num_recommendations_widget.value
    recommendations = recommend(location, category, num_recommendations, distance, uid )
    update_folium_map(recommendations)
        
location_widget = widgets.Text(
    description = 'Your location:',
    placeholder = 'Enter here',
)
location_widget.layout.width = '400px'

uid_widget = widgets.Text(
    description = 'User ID:',
    placeholder = 'Enter here',
)
uid_widget.layout.width = '400px'

category_widget = widgets.Dropdown(
    description = 'Categoty:',
    options = categories_to_show,
    value=categories_to_show[0],
)
category_widget.layout.width = '400px'

distance_widget = widgets.FloatSlider(
    description = 'Distance:',
    min = 0,
    max = 10,
    step = 0.5,
    value=5,
)
distance_widget.layout.width = '400px'

num_recommendations_widget = widgets.IntSlider(
    description="# of recommendations:",
    min = 1,
    max = 20,
    value =5
)
num_recommendations_widget.layout.width = '400px'

location_widget.observe(update_recommendations,'value')
uid_widget.observe(update_recommendations,'value')
category_widget.observe(update_recommendations,'value')
distance_widget.observe(update_recommendations,'value')
num_recommendations_widget.observe(update_recommendations,'value')

initial_map = folium.Map(location=[39.9526, -75.1652],zoom_start=13)

def update_folium_map(recommendations):
    clear_output()
    initial_map = folium.Map(location=[39.9526, -75.1652],zoom_start=13)

    for bid, recommendation in recommendations.items():
        lat = recommendation[1]
        lon = recommendation[2]
        name = recommendation[3]
        stars = recommendation[4]
        popup_text = f"Name: {name}<br>Stars: {stars}"
        folium.Marker(
            location = [lat,lon],
            popup=popup_text,
        ).add_to(initial_map)
    display(location_widget, uid_widget, category_widget, distance_widget,num_recommendations_widget)
    display(initial_map)

def show_ui():
    display(location_widget, uid_widget, category_widget, distance_widget,num_recommendations_widget)
    display(initial_map)
    












