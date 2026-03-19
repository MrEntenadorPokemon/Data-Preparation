import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. Función para estandarizar la longitud de las trayectorias ---
def remuestrear_trayectoria(x, y, num_puntos=60):
    """Interpola la trayectoria para que tenga exactamente 'num_puntos'."""
    dist = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
    if dist[-1] == 0: 
        return np.zeros(num_puntos), np.zeros(num_puntos)
    
    dist = dist / dist[-1]
    alpha = np.linspace(0, 1, num_puntos)
    x_resampled = np.interp(alpha, dist, x)
    y_resampled = np.interp(alpha, dist, y)
    
    return x_resampled, y_resampled

# --- 2. Función NUEVA: Alineación geométrica ---
def alinear_por_centroide_y_angulo(x, y):
    """Desplaza la trayectoria a su centroide y la rota basándose en el ángulo de inicio."""
    # Encontrar el Centroide
    centroide_x = np.mean(x)
    centroide_y = np.mean(y)
    
    # Trasladar todos los puntos (el centroide ahora es 0,0)
    x_centrado = x - centroide_x
    y_centrado = y - centroide_y
    
    # Calcular el ángulo del primer punto respecto al centroide
    angulo_inicio = np.arctan2(y_centrado[0], x_centrado[0])
    
    # Rotar la trayectoria para que todas inicien apuntando hacia el mismo ángulo (0 grados)
    x_rotado = x_centrado * np.cos(-angulo_inicio) - y_centrado * np.sin(-angulo_inicio)
    y_rotado = x_centrado * np.sin(-angulo_inicio) + y_centrado * np.cos(-angulo_inicio)
    
    return x_rotado, y_rotado

# --- 3. Carga, transformación y agrupación de datos ---
directorio_principal = Path('C:/Users/pok/Downloads/users_01_to_10') 
carpeta_salida = Path('promedios_individuales_alineados')
carpeta_salida.mkdir(exist_ok=True) 

datos_por_gesto = {i: {'x': [], 'y': [], 'archivos': []} for i in range(1, 9)}

print("Cargando, remuestreando y ALINEANDO archivos de todos los usuarios...")
for carpeta_usuario in directorio_principal.iterdir():
    if carpeta_usuario.is_dir():
        for archivo_csv in carpeta_usuario.glob('*.csv'):
            partes = archivo_csv.stem.split('_')
            try:
                gesto_id = int(partes[1])
            except ValueError:
                continue
                
            try:
                df = pd.read_csv(archivo_csv)
            except Exception as e:
                print(f"Error cargando {archivo_csv.name}: {e}")
                continue

            if len(df) > 2:
                # Paso A: Igualar la cantidad de puntos
                x_res, y_res = remuestrear_trayectoria(df['x'].values, df['y'].values, num_puntos=60)
                
                # Paso B: Centrar y rotar la trayectoria
                x_alineado, y_alineado = alinear_por_centroide_y_angulo(x_res, y_res)
                
                # Guardamos los datos ya procesados
                datos_por_gesto[gesto_id]['x'].append(x_alineado)
                datos_por_gesto[gesto_id]['y'].append(y_alineado)
                datos_por_gesto[gesto_id]['archivos'].append(archivo_csv.name)

# --- 4. Filtrado y generación de 8 GRÁFICAS INDIVIDUALES ---
print("\nFiltrando anomalías y generando gráficas por gesto alineado...")

for gesto_id, datos in datos_por_gesto.items():
    if not datos['x']:
        print(f"No se encontraron datos válidos para el Gesto {gesto_id}. Saltando...")
        continue
        
    todas_x = np.array(datos['x'])
    todas_y = np.array(datos['y'])
    
    # Calcular la mediana como estándar robusto
    mediana_x = np.median(todas_x, axis=0)
    mediana_y = np.median(todas_y, axis=0)
    
    # Calcular distancias y definir umbral de "basura"
    distancias = np.mean(np.sqrt((todas_x - mediana_x)**2 + (todas_y - mediana_y)**2), axis=1)
    umbral = np.median(distancias) + (1.0 * np.std(distancias)) 
    indices_limpios = np.where(distancias <= umbral)[0]
    
    total_muestras = len(distancias)
    muestras_limpias = len(indices_limpios)
    basura_descartada = total_muestras - muestras_limpias
    
    print(f"Gesto {gesto_id} procesado: {total_muestras} muestras totales | {muestras_limpias} válidas | {basura_descartada} descartadas.")
    
    # Calcular el promedio final SOLO con datos limpios
    x_promedio = np.mean(todas_x[indices_limpios], axis=0)
    y_promedio = np.mean(todas_y[indices_limpios], axis=0)
    
    # --- 5. Crear la gráfica visualmente enriquecida ---
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor('#fdfdfd') 

    # Graficar las muestras limpias de fondo (muy tenues)
    for idx in indices_limpios:
        ax.plot(todas_x[idx], todas_y[idx], color='gray', alpha=0.08, zorder=1, linewidth=0.8)
        
    # Preparar el degradado de color y flechas para la trayectoria promedio
    t = np.linspace(0, 1, len(x_promedio[:-1]))
    dx = np.diff(x_promedio)
    dy = np.diff(y_promedio)
    
    # Graficar la trayectoria promedio
    quiver = ax.quiver(
        x_promedio[:-1], y_promedio[:-1], dx, dy, t, 
        angles='xy', scale_units='xy', scale=1, 
        cmap='plasma', width=0.012, headwidth=4, headlength=5, zorder=3
    )
    
    # Marcar puntos críticos: INICIO (Verde) y FIN (Fucsia)
    ax.scatter(x_promedio[0], y_promedio[0], c='lime', s=180, marker='o', edgecolors='black', linewidth=1.5, zorder=5, label='Inicio Trazo')
    ax.scatter(x_promedio[-1], y_promedio[-1], c='fuchsia', s=180, marker='X', edgecolors='black', linewidth=1.5, zorder=5, label='Fin Trazo')
    
    # Formato de la gráfica
    ax.set_title(f'Gesto {gesto_id} - Promedio Centrado y Alineado\n({muestras_limpias} muestras válidas)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Coordenada X (Respecto al Centroide)', fontsize=11)
    ax.set_ylabel('Coordenada Y (Respecto al Centroide)', fontsize=11)
    
    # Dibujar líneas guía cruzando por el nuevo origen (0,0)
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.3, zorder=0)
    ax.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.3, zorder=0)
    
    ax.grid(True, linestyle=':', alpha=0.6, zorder=0)
    ax.axis('equal') 
    ax.legend(loc='best', framealpha=0.9, shadow=True)
    
    # Guardar la gráfica
    ruta_salida = carpeta_salida / f'promedio_gesto_{gesto_id:02d}_alineado.png'
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=160, bbox_inches='tight') 
    plt.close(fig) 

print(f"\n¡Proceso finalizado! Revisa las imágenes en la carpeta '{carpeta_salida.absolute()}'")