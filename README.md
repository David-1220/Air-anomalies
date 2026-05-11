# Air-anomalies
Proyecto 2 computo paralelo y distribuido, SISTEMA DE VIGILANCIA: ANOMALIAS DE ALTITUD EN VUELOS con sensores en Argentina

python3 -m venv .venv
. ./.venv/Scripts/activate
pip install -r requirements.txt
mpiexec -n 4 python main.py
streamlit run dashboard.py
