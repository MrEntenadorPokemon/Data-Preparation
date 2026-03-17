import pandas as pd
import numpy as np
import os
import glob

base_path = 'C:/Users/Pok/Downloads/users_01_to_10' 

all_rows = []

files = glob.glob(os.path.join(base_path, 'user_*', 'gesture_*.csv'))

print(f"Se encontraron {len(files)} archivos para procesar.")

for file_path in files:
    folder_name = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    
    user_id = folder_name.split('_')[1]
    parts = file_name.replace('.csv', '').split('_')
    gesture_id = parts[1]
    sample_id = parts[3]
    
    df = pd.read_csv(file_path)
    if df.empty: continue

    x_coords = df['x']
    y_coords = df['y']
    
    width_bbox = x_coords.max() - x_coords.min()
    height_bbox = y_coords.max() - y_coords.min()
    
    start_pt = (x_coords.iloc[0], y_coords.iloc[0])
    end_pt = (x_coords.iloc[-1], y_coords.iloc[-1])
    net_dist = np.sqrt((end_pt[0] - start_pt[0])**2 + (end_pt[1] - start_pt[1])**2)

    all_rows.append({
        'user_label': user_id,
        'gesture_label': gesture_id,
        'sample_id': sample_id,
        'points_count': len(df),
        'width_bbox': width_bbox,
        'height_bbox': height_bbox,
        'aspect_ratio': height_bbox / width_bbox if width_bbox != 0 else 0,
        'net_distance': net_dist,
        'mean_pressure': df['finger_pressure'].mean(),
        'max_pressure': df['finger_pressure'].max(),
        'mean_finger_w': df['width'].mean(),
        'mean_finger_h': df['height'].mean()
    })

final_df = pd.DataFrame(all_rows)

final_df = final_df.drop_duplicates()

final_df.to_csv('entrenamiento_gestos_final.csv', index=False)
print("¡Archivo 'entrenamiento_gestos_final.csv' generado con éxito!")