from flask import Flask, render_template, request, jsonify
from geometry_msgs.msg import Point32
from utils import gps_to_local_xy
import threading
import requests
from shapely.geometry import Polygon, Point

 
template_dir = 'templates'
static_dir = 'templates/static/'
print('template =',static_dir)
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

idx = 0
path = None

live_lat = None
live_lon = None
origin_lat = None
origin_lon = None
@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/clear_path', methods =['POST'])
# def clear_path():
#     global path
#     path = None
#     print('Path cleared')
#     return jsonify({'status': 'success'}), 200

@app.route('/generate_grid', methods=['POST'])
def generate_grid():
    data = request.get_json()
    polygon_pts = data['polygon']            # [{lat, lng}, …]
    spacing_x = float(data['spacingX'])      # metros no eixo Leste–Oeste
    spacing_y = float(data['spacingY'])      # metros no eixo Norte–Sul
    species_cfg = data['species']            # {ervas, arbustos, arvores}

    # Sequência repetitiva de espécies
    seq_species  = [species_cfg['ervas'],
                    species_cfg['arbustos'],
                    species_cfg['arvores'],
                    species_cfg['arbustos'],
                    species_cfg['ervas']]
    seq_category = ['Ervas','Arbustos','Árvores','Arbustos','Ervas']

    # Construir polígono Shapely (x=lng, y=lat)
    poly = Polygon([(pt['lng'], pt['lat']) for pt in polygon_pts])
    min_lng, min_lat, max_lng, max_lat = poly.bounds

    # Converter metros em graus (aprox. 1m ≃ 0.00001°)
    d_lng = spacing_x * 0.00001
    d_lat = spacing_y * 0.00001

    waypoints = []
    idx = 0

    # Varre de norte para sul e de oeste para leste
    y = max_lat
    while y >= min_lat:
        x = min_lng
        while x <= max_lng:
            if poly.contains(Point(x, y)):
                waypoints.append({
                    'lat': y,
                    'lng': x,
                    'species': seq_species[idx % len(seq_species)],
                    'category': seq_category[idx % len(seq_category)]
                })
                idx += 1
            x += d_lng
        y -= d_lat

    return jsonify({'waypoints': waypoints}), 200

@app.route('/submit_points', methods=['POST'])
def submit_points():
    global path, idx, origin_lat, origin_lon
    data = request.get_json()
    # print("Pontos recebidos:", data)
    
    local_path = []
    if (idx == 0): #pega apenas a posição inicial do drone
        origin_lat = data['droneLatLng']['lat'] 
        origin_lon = data['droneLatLng']['lng']
        idx = 1
        
    # print(origin_lat)
    
    for point in data['payload']:
        x, y = gps_to_local_xy(point['lat'], point['lng'], origin_lat, origin_lon)
        local_path.append({'name': point['species'], 'x':x,'y':y,'z':0.0})
    
    # path = tsp_solver(local_path)
    path = local_path
    print(f"→ Path recebido: {path}")
    # print(path)
    return jsonify({'status': 'success'}), 200

@app.route('/get_path', methods=['GET'])
def get_path():
    global path
    if path is None:
        return jsonify({'available': False})
    else:
        return jsonify({
            'available': True,
            'path': path
        })
# @app.route('/submit_polygon', methods = ['POST'])
# def submit_polygon():
#     data = request.get_json()
#     print(data)
#     origin_lat = data['polygon'][0]['lat']
#     origin_lon =  data['polygon'][0]['lng']
#     points = []
#     for point in data['polygon']:
#         x,y = gps_to_local_xy(point['lat'], point['lng'], origin_lat, origin_lon)
#         points.append((x,y))
#     print(points)
#     thread = threading.Thread(target=evolution_loop, args=(points, data['species']))
#     thread.start()
#     return jsonify({'status': 'success'}), 200

if __name__ == '__main__':
    # rospy.init_node("drone_planter", anonymous=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
