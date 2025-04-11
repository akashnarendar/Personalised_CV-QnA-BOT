from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'Narendar',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='cv_faiss_index_rebuilder',
    default_args=default_args,
    description='Rebuilds the FAISS index from updated CV',
    schedule_interval='@daily',
    start_date=datetime(2025, 4, 11),
    catchup=False,
    tags=['faiss', 'index', 'cv'],
) as dag:

    build_index = BashOperator(
        task_id='rebuild_faiss_index',
        bash_command='CV_PATH="/opt/airflow/data/docs/Narendar_Punithan_CV_final_word.txt" python /opt/airflow/scripts/build_index.py',
    )

    build_index

