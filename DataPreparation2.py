import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. Función para estandarizar la longitud de las trayectorias ---
def remuestrear_trayectoria(x, y, num_puntos=50):
    """Interpola la trayectoria para que tenga exactamente 'num_puntos'."""
    dist = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
    if dist[-1] == 0: 
        return np.zeros(num_puntos), np.zeros(num_puntos)
    
    dist = dist / dist[-1]
    alpha = np.linspace(0, 1, num_puntos)
    x_resampled = np.interp(alpha, dist, x)
    y_resampled = np.interp(alpha, dist, y)
    
    return x_resampled, y_resampled

# --- 2. Carga y agrupación de datos ---
# Directorio base (asegúrate de que esta carpeta exista y contenga los user_01...user_10)
directorio_principal = Path('C:/Users/pok/Downloads/users_01_to_10') 
carpeta_salida = Path('promedios_individuales_color')
carpeta_salida.mkdir(exist_ok=True) # Crea la carpeta si no existe

datos_por_gesto = {i: {'x': [], 'y': [], 'archivos': []} for i in range(1, 9)}

print("Cargando y remuestreando archivos de todos los usuarios...")
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
                # Usamos 60 puntos para una mejor resolución en gestos complejos
                x_res, y_res = remuestrear_trayectoria(df['x'].values, df['y'].values, num_puntos=60)
                datos_por_gesto[gesto_id]['x'].append(x_res)
                datos_por_gesto[gesto_id]['y'].append(y_res)
                datos_por_gesto[gesto_id]['archivos'].append(archivo_csv.name)

# --- 3. Filtrado y generación de 8 GRÁFICAS INDIVIDUALES MEJORADAS ---
print("\nFiltrando anomalías y generando gráficas por gesto...")

for gesto_id, datos in datos_por_gesto.items():
    if not datos['x']:
        print(f"No se encontraron datos válidos para el Gesto {gesto_id}. Saltando...")
        continue
        
    todas_x = np.array(datos['x'])
    todas_y = np.array(datos['y'])
    
    # Calcular la mediana para usarla como estándar robusto
    mediana_x = np.median(todas_x, axis=0)
    mediana_y = np.median(todas_y, axis=0)
    
    # Calcular distancias y definir umbral de "basura" (más estricto: mediana + 1 std)
    distancias = np.mean(np.sqrt((todas_x - mediana_x)**2 + (todas_y - mediana_y)**2), axis=1)
    umbral = np.median(distancias) + (1.0 * np.std(distancias)) # Filtro estándar
    indices_limpios = np.where(distancias <= umbral)[0]
    
    total_muestras = len(distancias)
    muestras_limpias = len(indices_limpios)
    basura_descartada = total_muestras - muestras_limpias
    
    print(f"Gesto {gesto_id} procesado: {total_muestras} muestras totales | {muestras_limpias} válidas | {basura_descartada} descartadas.")
    
    # Calcular el promedio final SOLO con datos limpios
    x_promedio = np.mean(todas_x[indices_limpios], axis=0)
    y_promedio = np.mean(todas_y[indices_limpios], axis=0)
    
    # --- 4. Crear la gráfica MEJORADA para este gesto específico ---
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor('#fdfdfd') # Un fondo ligeramente grisáceo muy tenue para que los blancos/amarillos resalten

    # 4.1. Graficar las muestras limpias de fondo (SÚPER TENUES) para contexto
    for idx in indices_limpios:
        ax.plot(todas_x[idx], todas_y[idx], color='gray', alpha=0.08, zorder=1, linewidth=0.8)
        
    # 4.2. Preparar el degradado de color para las flechas promedio
    # Generamos un array de 0 a 1 que representa el "tiempo"
    t = np.linspace(0, 1, len(x_promedio[:-1]))
    
    # Diferenciales para las flechas
    dx = np.diff(x_promedio)
    dy = np.diff(y_promedio)
    
    # 4.3. Graficar la TRAYECTORIA PROMEDIO usando flechas con DEGRADADO
    # Usamos cmap='plasma' que va de morado -> naranja -> amarillo brillante (buen contraste)
    quiver = ax.quiver(
        x_promedio[:-1], y_promedio[:-1], dx, dy, t, # t define el color de cada flecha
        angles='xy', scale_units='xy', scale=1, 
        cmap='plasma', width=0.012, headwidth=4, headlength=5, zorder=3
    )
    
    # 4.4. Marcar puntos críticos con ALTO CONTRASTE
    # Punto de INICIO: Círculo Verde Lima Grande
    ax.scatter(x_promedio[0], y_promedio[0], c='lime', s=180, marker='o', edgecolors='black', linewidth=1.5, zorder=5, label='Inicio Trazo')
    
    # Punto de FIN: X Fucsia/Roja Grande
    ax.scatter(x_promedio[-1], y_promedio[-1], c='fuchsia', s=180, marker='X', edgecolors='black', linewidth=1.5, zorder=5, label='Fin Trazo')
    
    # 4.5. Formato de la gráfica para lectura fácil
    ax.set_title(f'Gesto {gesto_id} - Trayectoria Promedio Limpia\n({muestras_limpias} muestras, degradado indica paso del tiempo)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Coordenada X (normalizada por origen)', fontsize=11)
    ax.set_ylabel('Coordenada Y (normalizada por origen)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.axis('equal') # Mantener la proporción
    
    # Añadir barra de color sutil para indicar tiempo (opcional, si quieres más claridad)
    # cbar = fig.colorbar(quiver, ax=ax, ticks=[0, 1], shrink=0.5, location='bottom', pad=0.1)
    # cbar.ax.set_xticklabels(['Inicio', 'Fin'])
    
    ax.legend(loc='best', framealpha=0.9, shadow=True)
    
    # Guardar la gráfica en la nueva carpeta
    ruta_salida = carpeta_salida / f'promedio_gesto_{gesto_id:02d}_color.png'
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=160, bbox_inches='tight') # dpi alto y bbox tight para evitar cortes
    plt.close(fig) # Importante: cerrar para que no se congele la computadora

print(f"\n¡Proceso finalizado! Revisa las 8 imágenes mejoradas en la carpeta '{carpeta_salida.absolute()}'")