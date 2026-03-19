import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURACIÓN Y RUTAS ---
# RECUERDA CAMBIAR ESTA RUTA A TU CARPETA LOCAL DESCOMPRIMIDA
dataset_path = 'C:/Users/pok/Downloads/users_01_to_10' 

# Carpetas de salida para organizar
csv_output_dir = 'datos_procesados_final'
plot_output_dir = 'graficas_para_reporte'

for d in [csv_output_dir, plot_output_dir]:
    if not os.path.exists(d): os.makedirs(d)

extracted_features = []
files = glob.glob(os.path.join(dataset_path, 'user_*', 'gesture_*.csv'))
print(f"--- Iniciando procesamiento de {len(files)} archivos crudos ---")

# Diccionario para guardar ejemplos para las gráficas de trayectoria
muestras_ejemplo = {} 

# --- 2. BUCLE PRINCIPAL: PROCESAMIENTO Y FEATURE ENGINEERING ---
for file_path in files:
    folder_name = os.path.basename(os.path.dirname(file_path)) # user_01
    file_name = os.path.basename(file_path) # gesture_01_sample_01.csv
    
    user_id = folder_name.split('_')[1]
    gesture_parts = file_name.replace('.csv', '').split('_')
    gesture_id = gesture_parts[1]
    sample_id = gesture_parts[3]
    
    df = pd.read_csv(file_path)
    if df.empty: continue

    # Guardar ejemplos para el gráfico de trayectoria
    if gesture_id not in muestras_ejemplo:
        muestras_ejemplo[gesture_id] = df.copy()

    # --- CÁLCULO DE CARACTERÍSTICAS FIJAS ---
    # 1. Calidad de Datos (Trash Detection Feature)
    num_pts = len(df)
    
    # 2. Características Espaciales (BBox & Aspect Ratio)
    xmin, xmax = df['x'].min(), df['x'].max()
    ymin, ymax = df['y'].min(), df['y'].max()
    bbox_w = xmax - xmin
    bbox_h = ymax - ymin
    
    # 3. Características de Trayectoria (Distancia Euclidiana Neta)
    xstart, ystart = df['x'].iloc[0], df['y'].iloc[0]
    xend, yend = df['x'].iloc[-1], df['y'].iloc[-1]
    dist_neta = np.sqrt((xend - xstart)**2 + (yend - ystart)**2)

    # --- GUARDAR EN VECTOR ---
    extracted_features.append({
        'user_id_raw': user_id,
        'gesture_id_raw': gesture_id,
        'sample_id_raw': sample_id,
        # Features para clasificación
        'num_points': num_pts,
        'bbox_width': bbox_w,
        'bbox_height': bbox_h,
        'aspect_ratio': bbox_h / bbox_w if bbox_w != 0 else 0,
        'net_distance': dist_neta,
        'avg_pressure': df['finger_pressure'].mean(),
        'max_pressure': df['finger_pressure'].max(),
        'avg_finger_w': df['width'].mean(),
        'avg_finger_h': df['height'].mean()
    })

# --- 3. CREACIÓN Y ESTRUCTURACIÓN DEL DATASET INICIAL ---
main_df = pd.DataFrame(extracted_features)

# Creación de etiquetas finales (Integers) para los modelos
main_df['user_label'] = main_df['user_id_raw'].astype(int)
main_df['gesture_label'] = main_df['gesture_id_raw'].astype(int)

print(f"Dataset inicial creado con {len(main_df)} muestras.")

# --- 4. LIMPIEZA PROFUNDA: SEPARACIÓN DE BASURA Y DUPLICADOS ---

# A. Identificación de "Archivos Basura" (Criterio: menos de 10 puntos temporales)
# Esto responde a la pregunta de problemas encontrados.
print("\n--- Identificando y separando archivos basura ---")
threshold_puntos = 10
trash_mask = main_df['num_points'] < threshold_puntos
trash_df = main_df[trash_mask].copy()
clean_df = main_df[~trash_mask].copy()

print(f"Muestras basura identificadas (menos de {threshold_puntos} pts): {len(trash_df)}")
print(f"Muestras limpias restantes: {len(clean_df)}")

# B. Eliminación de Duplicados en el set de entrenamiento
# Usamos las características como criterio de unicidad
features_to_check = ['bbox_width', 'bbox_height', 'aspect_ratio', 'net_distance', 'avg_pressure']
print(f"\n--- Eliminando duplicados exactos en características principales ---")
len_antes = len(clean_df)
clean_df_unique = clean_df.drop_duplicates(subset=features_to_check, keep='first')
len_despues = len(clean_df_unique)
print(f"Duplicados eliminados: {len_antes - len_despues}")

# --- 5. GUARDADO DE ARCHIVOS CSV FINALES (PRODUCTO PRINCIPAL) ---
print("\n--- Guardando archivos CSV finales ---")

# Archivo de Entrenamiento Limpio (Lo que entregarás)
clean_df_unique.to_csv(f'{csv_output_dir}/entrenamiento_gestos_final.csv', index=False)
# Archivo de Basura Separada (Para tu reporte/evidencia)
trash_df.to_csv(f'{csv_output_dir}/archivos_basura_limpiados.csv', index=False)

print(f"Archivos guardados en '{csv_output_dir}/'.")

# --- 6. GENERACIÓN AUTOMÁTICA DE GRÁFICAS PARA EL REPORTE ---
print("\n--- Generando gráficas para el reporte... ---")
sns.set_style("whitegrid") # Estilo limpio

# G1: Trayectorias de Ejemplo (Determinar estado de muestras)
# Como en image_3411a4.png, pero agrupadas.
plt.figure(figsize=(16, 8))
for i, (gest_id, data) in enumerate(sorted(muestras_ejemplo.items())):
    plt.subplot(2, 4, i+1)
    plt.plot(data['x'], data['y'], marker='o', markersize=3, alpha=0.7)
    plt.title(f"Visualización: Gesto {gest_id}")
    plt.gca().invert_yaxis() # Invertir para vista de tableta (0,0 es arriba)
plt.tight_layout()
plt.savefig(f'{plot_output_dir}/01_trayectorias_ejemplo.png')
plt.close()

# G2: Balance de Clases: Gestos
# Responde "¿Están bien representadas y balanceadas?"
plt.figure(figsize=(10, 6))
sns.countplot(data=clean_df_unique, x='gesture_label', palette='viridis')
plt.title('Distribución de Clases por Gesto (Datos Limpios)')
plt.xlabel('ID del Gesto')
plt.ylabel('Cantidad de Muestras')
plt.savefig(f'{plot_output_dir}/02_balance_clases_gestos.png')
plt.close()

# G3: Balance de Clases: Usuarios (Requerido para Sección B)
plt.figure(figsize=(12, 6))
sns.countplot(data=clean_df_unique, x='user_label', palette='Set2')
plt.title('Distribución de Muestras por Usuario')
plt.xlabel('ID del Usuario')
plt.ylabel('Cantidad de Muestras')
plt.savefig(f'{plot_output_dir}/03_balance_clases_usuarios.png')
plt.close()

# G4: Análisis de Criterios (Boxplots como en image_34119c.png)
# Responde "¿Qué representación utilizarías?" mostrando diferenciación.
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=clean_df_unique, x='gesture_label', y='net_distance', palette='coolwarm')
plt.title('Diferenciación de Gestos por Distancia Neta')
plt.subplot(1, 2, 2)
sns.boxplot(data=clean_df_unique, x='gesture_label', y='aspect_ratio', palette='coolwarm')
plt.title('Diferenciación de Gestos por Relación de Aspecto')
plt.savefig(f'{plot_output_dir}/04_analisis_features.png')
plt.close()

print(f"--- ¡Listo! Proceso completo ---")
print(f"Revisa las carpetas '{csv_output_dir}/' y '{plot_output_dir}/'.")