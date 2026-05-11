import streamlit as st
import pandas as pd
import time
import os

st.set_page_config(page_title="PROYECTO 2 - EQUIPO 1", layout="wide")

def load_data():
    if os.path.exists("resultados_finales.csv"):
        try:
            df = pd.read_csv("resultados_finales.csv")
            df_map = df.dropna(subset=['lat', 'lng']).copy()
            df_map = df_map.rename(columns={'lng': 'lon'})
            return df_map
        except: return None
    return None

st.title(" Radar Inteligente: Detección de Anomalías Geoespaciales")
st.markdown("**Arquitectura HPC: MPI + CUDA | Variable de Control: Presión Barométrica (IoT)**")
st.markdown("---")

placeholder = st.empty()

while True:
    df = load_data()
    if df is not None and not df.empty:
        with placeholder.container():
            # 1. KPIs de Control
            k1, k2, k3, k4 = st.columns(4)
            
            p_base = df['presion_servicio'].iloc[0] if 'presion_servicio' in df.columns else 1013.25
            total_anomalias = df['es_anomalia'].sum() if 'es_anomalia' in df.columns else 0
            
    # Filtro para la distancia de 1,500 km
            df_distancia= df[df['distancia_al_sensor_km'] < 1500]
            vuelos_cerca = len(df_distancia)
            
            k1.metric("Vuelos en Radar", f"{len(df):,}")
            k2.metric("En distancia (<1500km)", f"{vuelos_cerca}")
            k3.metric("Referencia IoT (hPa)", f"{p_base:.2f}")
            k4.metric("Alertas de Altitud", f"{int(total_anomalias)}", delta="Análisis GPU", delta_color="inverse")

    #Visual
            col_map, col_graph = st.columns([2, 1])
            with col_map:
                st.subheader(" Monitoreo Global y Proximidad")
                st.map(df[['lat', 'lon']].head(200))
            
            with col_graph:
                st.subheader(" Nivel de rareza De Altitud (Vuelos Cercanos)")
                if 'z_score' in df_distancia.columns:
    # Grafica de vuelos dentro del filtro de la distancia
                    if not df_distancia.empty:
                        st.bar_chart(df_distancia['z_score'].head(50))
                    else:
                        st.warning("No hay aeronaves detectadas en el radio de 1,500km.")

            st.subheader(" Detalle de Aeronaves en Zona de Influencia")
            cols_show = ['flight_iata', 'distancia_al_sensor_km', 'alt', 'speed', 'z_score', 'es_anomalia']
    # Tabla de datos con los vuelos dentro del filtro de la distancia 
            df_show = df_distancia[cols_show].sort_values(by='distancia_al_sensor_km').head(50)

            df_show = df_show.rename(columns={'z_score': 'Nivel de rareza'})
            st.dataframe(df_show, width='stretch')
            
            st.info(f"Actualización: {time.strftime('%H:%M:%S')}. La gráfica de Z-Score solo muestra aeronaves dentro del radio de influencia.")
            
    time.sleep(5)