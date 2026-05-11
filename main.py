from mpi4py import MPI
import asyncio
import pandas as pd
import numpy as np
from ingest import get_all_data
from processing import clean_data, gpu_analysis

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #  Nodo Maestro se encarga de la ingestión y limpieza de datos
    if rank == 0:
        print("--- Iniciando Pipeline Geoespacial y MPI (Equipo 1) ---")
        try:
            raw_data = asyncio.run(get_all_data())
            df_w, df_f = clean_data(raw_data.get("weather"), raw_data.get("flights"))
            
            if df_f.empty:
                print(" ERROR: No hay datos de vuelos disponibles.")
                comm.Abort()

            # Presión de Argentina para ajustar la sensibilidad de la detección de anomalías
            valor_base = df_w['Presion'].iloc[-1] if not df_w.empty else 1013.25
            
            # dividimos el DataFrame de vuelos en partes iguales para cada nodo
            chunks = np.array_split(df_f, size)
            cols = df_f.columns.tolist()
        except Exception as e:
            print(f" Fallo en Nodo Maestro: {e}")
            comm.Abort()
    else:
        chunks, cols, valor_base = None, None, None

    # Distribuimos los datos a los nodos trabajadores
    my_chunk_raw = comm.scatter(chunks, root=0)
    cols = comm.bcast(cols, root=0)
    valor_base = comm.bcast(valor_base, root=0)

    my_chunk = pd.DataFrame(my_chunk_raw, columns=cols) if isinstance(my_chunk_raw, np.ndarray) else my_chunk_raw

    # Cada núcleo usa la RTX 3050 para calcular distancias (Haversine) y anomalías
    local_results = gpu_analysis(my_chunk, valor_base)

    #  Recolectamos los resultados de todos los nodos al maestro
    all_results = comm.gather(local_results, root=0)

    # Reporte final en el nodo maestro
    if rank == 0:
        print("\n" + "="*60)
        print("--- SISTEMA DE VIGILANCIA: ANOMALIAS DE ALTITUD (EQUIPO 1) ---")
        print("="*60)
        
        # Unificación de resultados de los nodos MPI
        vuelos_totales = pd.concat([pd.DataFrame(r['data']) for r in all_results if 'data' in r])
        total_anomalias = int(vuelos_totales['es_anomalia'].sum())
        
        if 'distancia_al_sensor_km' in vuelos_totales.columns:
            vuelos_cerca = len(vuelos_totales[vuelos_totales['distancia_al_sensor_km'] < 1500])
        else:
            vuelos_cerca = 0
            
        print(f" -> Aeronaves Globales en Radar: {len(vuelos_totales)}")
        print(f" -> Aeronaves en Geocerca de Influencia (<1500km): {vuelos_cerca}")
        print(f" -> Presion Atmosferica Local (ThingSpeak): {valor_base:.2f} hPa")
        print(f" -> Alertas por Maniobras de Altitud Atipicas: {total_anomalias}")
        
        if total_anomalias > 0:
            print(f" [AVISO] Se detectaron {total_anomalias} vuelos con perfiles de altitud fuera de rango.")
        else:
            print(" [ESTADO] Operaciones Normales: Perfiles de vuelo estables.")
        
        print("="*60)
        
        # Exportación para el Dashboard de Streamlit
        vuelos_totales['presion_servicio'] = valor_base 
        vuelos_totales.to_csv("resultados_finales.csv", index=False)
        print("\n[OK] Pipeline finalizado. Resultados exportados a CSV.")

if __name__ == "__main__":
    main()