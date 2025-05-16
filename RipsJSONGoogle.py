import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster
from math import radians, cos, sin, sqrt, atan2
import json
import requests
import copy

GOOGLE_API_KEY = 'YOU_GOOGLE_API_KEY'

# --- Load EV charging stations ---
stations_df = pd.read_csv("TexasWithAvailaOSM.csv")
stations_df = stations_df.head(200)
station_locations = list(zip(stations_df['latitude'], stations_df['longitude']))
available_batteries = stations_df['availability'].tolist()
station_ids = stations_df['id'].tolist()

# --- Load car locations ---
car_locations_df = pd.read_csv("car_locations.csv")
car_locations = list(zip(car_locations_df['latitude'], car_locations_df['longitude']))

def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def get_google_distance(origin, destination):
    url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin[0]},{origin[1]}&destinations={destination[0]},{destination[1]}&key={GOOGLE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data['rows'][0]['elements'][0]['status'] == 'OK':
        return data['rows'][0]['elements'][0]['distance']['value'] / 1000
    return None

def get_google_directions(origin, destination):
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin[0]},{origin[1]}&destination={destination[0]},{destination[1]}&key={GOOGLE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'OK':
        return decode_polyline(data['routes'][0]['overview_polyline']['points'])
    return []

def decode_polyline(polyline_str):
    import polyline
    return polyline.decode(polyline_str)

soc_percent = 15
range_km = (soc_percent / 100) * 500
rips_radius_km = range_km

# --- Filter stations with ≥20 batteries and keep original ID ---
filtered_stations = []
for idx, ((lat, lon), avail, sid) in enumerate(zip(station_locations, available_batteries, station_ids)):
    if avail >= 20:
        filtered_stations.append((lat, lon, avail, sid))

edges = []
for i, loc1 in enumerate(filtered_stations):
    for j, loc2 in enumerate(filtered_stations):
        if i < j and haversine((loc1[0], loc1[1]), (loc2[0], loc2[1])) <= rips_radius_km:
            edges.append(((loc1[0], loc1[1]), (loc2[0], loc2[1])))

midpoint = station_locations[0]
m = folium.Map(location=midpoint, zoom_start=8)

station_cluster = MarkerCluster(name="Charging Stations").add_to(m)
for lat, lon, avail, sid in filtered_stations:
    folium.CircleMarker(
        location=(lat, lon),
        radius=3,
        color='green',
        fill=True,
        fill_color='green',
        fill_opacity=0.7,
        tooltip=f"Station ID: {sid}, Batteries: {avail}"
    ).add_to(station_cluster)
    folium.map.Marker(
        [lat, lon],
        icon=folium.DivIcon(html=f"""<div style="font-size: 8pt; color: green;"><b>{sid}</b></div>""")
    ).add_to(m)

for lat, lon, avail, _ in filtered_stations:
    folium.Circle(
        location=(lat, lon),
        radius=rips_radius_km * 1000,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.3
    ).add_to(m)

reachable_counts = []
nearest_station_data = []

for car_idx, car in enumerate(car_locations):
    reachable = []
    for lat, lon, avail, sid in filtered_stations:
        if haversine(car, (lat, lon)) <= rips_radius_km:
            reachable.append({
                'location': (lat, lon),
                'batteries': avail,
                'station_id': sid
            })
    reachable_counts.append(len(reachable))
    
    if reachable:
        best_station = max(reachable, key=lambda x: x['batteries'])
        distance_km = get_google_distance(car, best_station['location'])
        route_coords = get_google_directions(car, best_station['location']) if distance_km else []

        nearest_station_data.append({
            'car_id': car_idx + 1,
            'station_id': best_station['station_id'],
            'station_location': best_station['location'],
            'available_batteries': best_station['batteries'],
            'distance_km': round(distance_km, 2) if distance_km else None,
            'route_coords': route_coords
        })
    else:
        nearest_station_data.append({
            'car_id': car_idx + 1,
            'station_id': None,
            'station_location': None,
            'available_batteries': 0,
            'distance_km': None,
            'route_coords': []
        })

nearest_station_data_no_route = copy.deepcopy(nearest_station_data)
for item in nearest_station_data_no_route:
    item.pop('route_coords', None)

with open("car_nearest_station.json", "w") as f:
    json.dump(nearest_station_data_no_route, f, indent=4)

with open("car_nearest_station.json", "r") as f:
    nearest_station_json = json.load(f)

car_icon_url = "https://cdn-icons-png.flaticon.com/512/5193/5193688.png"
for idx, car in enumerate(car_locations):
    icon = folium.CustomIcon(car_icon_url, icon_size=(30, 30))
    popup_html = f"<b>EV Car ID: {idx + 1}</b><br>"

    nearest_data = nearest_station_json[idx]
    if nearest_data['station_id'] is not None:
        popup_html += f"Nearest Station ID: {nearest_data['station_id']}<br>"
        popup_html += f"Available Batteries: {nearest_data['available_batteries']}<br>"
        popup_html += f"Google Distance: {nearest_data['distance_km']} km"
        tooltip_text = f"ID: {nearest_data['station_id']}, Distance: {nearest_data['distance_km']} km"

        if nearest_station_data[idx]['route_coords']:
            folium.PolyLine(nearest_station_data[idx]['route_coords'], color='purple', weight=4, opacity=0.7).add_to(m)
    else:
        popup_html += "No reachable station."
        tooltip_text = "No reachable station"

    folium.Marker(
        location=car,
        icon=icon,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=tooltip_text
    ).add_to(m)

    folium.map.Marker(
        [car[0], car[1]],
        icon=folium.DivIcon(html=f"""<div style="font-size: 8pt; color: red;"><b>{idx + 1}</b></div>""")
    ).add_to(m)

for p1, p2 in edges:
    folium.PolyLine(locations=[p1, p2], color='black', weight=3, opacity=0.6, tooltip="Rips Edge").add_to(m)

plt.figure(figsize=(12, 6))
x_labels = [f"Car {i+1}" for i in range(len(car_locations))]
plt.bar(x_labels, reachable_counts, color='skyblue', edgecolor='black')
plt.xlabel("EV Car ID")
plt.ylabel("Number of Reachable Charging Stations")
plt.title("Reachability of EV Cars to Stations (via Rips Disks)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("ev_reachability_bargraph.png")
plt.close()

m.save("rips_texas_map_with_cars.html")

station_details_json = []
for lat, lon, avail, sid in filtered_stations:
    station_details_json.append({
        "station_id": sid,
        "latitude": lat,
        "longitude": lon,
        "available_batteries": avail
    })

with open("station_details.json", "w") as f:
    json.dump(station_details_json, f, indent=4)

reachable_stations_json = []
for car_idx, count in enumerate(reachable_counts):
    reachable_stations_json.append({
        "car_id": car_idx + 1,
        "reachable_stations_count": count
    })

with open("car_reachable_stations.json", "w") as f:
    json.dump(reachable_stations_json, f, indent=4)
