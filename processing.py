import pandas as pd
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = np 

def clean_data(weather_raw, flights_raw):
    df_w = pd.DataFrame()
    if isinstance(weather_raw, dict) and 'feeds' in weather_raw:
        df_w = pd.DataFrame(weather_raw['feeds'])
        if 'field3' in df_w.columns:
            df_w['Presion'] = pd.to_numeric(df_w['field3'], errors='coerce')
            df_w = df_w.dropna(subset=['Presion'])
    
    df_f = pd.DataFrame()
    if isinstance(flights_raw, dict) and 'response' in flights_raw:
        df_f = pd.DataFrame(flights_raw['response'])
        for col in ['delay', 'speed', 'alt', 'lat', 'lng']:
            if col in df_f.columns:
                df_f[col] = pd.to_numeric(df_f[col], errors='coerce').fillna(0)
            else:
                df_f[col] = 0.0
    return df_w, df_f


# Coordenadas exactas de tu sensor en Argentina (Ej: Buenos Aires)
LAT_SENSOR = -42.7859055676
LON_SENSOR = -64.9964695031

# Función de análisis en la GPU usando CuPy para cálculos masivos de distancia y detección de anomalías
def gpu_analysis(df, presion_base):
    """Análisis Geoespacial y Detección de Anomalías de Altitud por Clima"""
    if df.empty: return {"data": df.to_dict()}
    
    try:
        # analizamos la ALTITUD (alt) para detectar maniobras evasivas
        altitudes = cp.array(df['alt'].astype(float).values)
        velocidades = cp.array(df['speed'].astype(float).values) # Solo para guardarla
        lat_vuelos = cp.array(df['lat'].astype(float).values)
        lon_vuelos = cp.array(df['lng'].astype(float).values)
        
        # 2. CÁLCULO MÁSIVO DE DISTANCIAS (Haversine Formula en CUDA)
        lat1, lon1 = cp.radians(LAT_SENSOR), cp.radians(LON_SENSOR)
        lat2, lon2 = cp.radians(lat_vuelos), cp.radians(lon_vuelos)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = cp.sin(dlat/2)**2 + cp.cos(lat1) * cp.cos(lat2) * cp.sin(dlon/2)**2
        c = 2 * cp.arcsin(cp.sqrt(a))
        radio_tierra = 6371 
        distancias_km = c * radio_tierra
        
        # 3. NIVEL DE RAREZA DE ALTITUD Y UMBRAL DINÁMICO
        mean_a = cp.mean(altitudes)
        std_a = cp.std(altitudes) + 1e-5
        z_scores = cp.abs((altitudes - mean_a) / std_a)
        
        # Ajustamos el umbral de detección según la presión atmosférica
        # si tnemos: Baja presión = Tormentas = Detectamos cambios de altitud atípicos en la proximidad del sensor
        if presion_base < 1010: 
            umbral = cp.where(distancias_km < 1500, 1.5, 2.5) 
        else: #si tenemos un Clima estable = Tolerancia normal de vuelo
            umbral = cp.where(distancias_km < 1500, 2.0, 2.5)
            
        anomalias = cp.where(z_scores > umbral, 1, 0)
        
        # Guardamos nuestros resultados
        df['distancia_al_sensor_km'] = cp.asnumpy(distancias_km)
        df['z_score'] = cp.asnumpy(z_scores)
        df['es_anomalia'] = cp.asnumpy(anomalias)
        df['presion_vuelo'] = cp.asnumpy(presion_base * cp.power((1 - 0.0000068753 * altitudes), 5.2559))
        
    except Exception as e:
        print(f"Error en Kernel GPU: {e}")
    
    return {"data": df.to_dict()}
    
