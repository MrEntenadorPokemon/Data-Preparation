import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def procesar_y_extraer_features(df, num_puntos=60):
    """Realiza la limpieza matemática y extrae el vector de features de una muestra cruda."""
    x = df['x'].values
    y = df['y'].values
    
    # 1. Remuestreo
    dist = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
    if dist[-1] == 0: return None
    dist_norm = dist / dist[-1]
    alpha = np.linspace(0, 1, num_puntos)
    x_res = np.interp(alpha, dist_norm, x)
    y_res = np.interp(alpha, dist_norm, y)
    
    # 2. Traslación al centroide y Rotación
    cx, cy = np.mean(x_res), np.mean(y_res)
    x_c, y_c = x_res - cx, y_res - cy
    angulo = np.arctan2(y_c[0], x_c[0])
    x_rot = x_c * np.cos(-angulo) - y_c * np.sin(-angulo)
    y_rot = x_c * np.sin(-angulo) + y_c * np.cos(-angulo)
    
    # 3. Cálculo de características extra estáticas
    features_extra = {
        'aspect_ratio': (np.max(x) - np.min(x)) / (np.max(y) - np.min(y) + 1e-6),
        'net_distance': np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2),
        'path_length': dist[-1],
        'avg_pressure': df['finger_pressure'].mean(),
        'max_pressure': df['finger_pressure'].max(),
        'avg_finger_w': df['width'].mean(),
        'avg_finger_h': df['height'].mean()
    }
    
    # Vectorizamos las coordenadas
    features_coords = {}
    for i in range(num_puntos):
        features_coords[f'x{i}'] = x_rot[i]
        features_coords[f'y{i}'] = y_rot[i]
        
    # Unimos todo en un solo diccionario
    return {**features_coords, **features_extra}

# --- EJECUCIÓN DEL PIPELINE ---
directorio = Path('C:/Users/pok/Downloads/users_01_to_10')
dataset_final = []

print("1. Extrayendo características y armando Dataset Tabular...")
for carpeta in directorio.iterdir():
    if carpeta.is_dir():
        for archivo in carpeta.glob('*.csv'):
            partes = archivo.stem.split('_')
            try: gesture_label = int(partes[1])
            except: continue
                
            df_crudo = pd.read_csv(archivo)
            if len(df_crudo) > 3:
                features = procesar_y_extraer_features(df_crudo)
                if features:
                    features['gesture_label'] = gesture_label # Nuestra variable Y
                    dataset_final.append(features)

# Convertir a DataFrame y eliminar duplicados (si los hubiera)
df_final = pd.DataFrame(dataset_final).drop_duplicates()

# Guardar el PRODUCTO PRINCIPAL (El archivo CSV)
archivo_salida = 'dataset_entrenamiento_gestos.csv'
df_final.to_csv(archivo_salida, index=False)
print(f"2. ¡Éxito! Archivo de entrenamiento guardado como: {archivo_salida} con {len(df_final)} muestras.")

# --- MODELADO: IMPLEMENTACIÓN DE RANDOM FOREST ---
print("\n3. Entrenando Modelo de Clasificación (Random Forest)...")
X = df_final.drop('gesture_label', axis=1)
y = df_final['gesture_label']

# Separamos el archivo general en datos de Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Instanciamos y entrenamos el algoritmo de bonificación
modelo_rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
modelo_rf.fit(X_train, y_train)

# Predicción y Evaluación
y_pred = modelo_rf.predict(X_test)
precision = accuracy_score(y_test, y_pred)

print(f"\nResultados del Algoritmo:")
print(f"Precisión Global (Accuracy): {precision * 100:.2f}%\n")
print(classification_report(y_test, y_pred))