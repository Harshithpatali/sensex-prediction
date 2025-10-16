from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import subprocess

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 10, 16),
    'retries': 1,
}

dag = DAG(
    'sensex_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
)

def run_ingest():
    subprocess.run(["python", "/opt/airflow/src/ingestion/step1_ingest.py"], check=True)

def run_clean():
    subprocess.run(["python", "/opt/airflow/src/cleaning/step2_clean.py"], check=True)

def run_features():
    subprocess.run(["python", "/opt/airflow/src/features/step3_features.py"], check=True)

def run_train():
    subprocess.run(["python", "/opt/airflow/src/model/step5_train_lstm.py"], check=True)

def run_predict():
    subprocess.run(["python", "/opt/airflow/src/model/step6_predict.py"], check=True)

t1 = PythonOperator(task_id='ingest_data', python_callable=run_ingest, dag=dag)
t2 = PythonOperator(task_id='clean_data', python_callable=run_clean, dag=dag)
t3 = PythonOperator(task_id='feature_engineering', python_callable=run_features, dag=dag)
t4 = PythonOperator(task_id='train_model', python_callable=run_train, dag=dag)
t5 = PythonOperator(task_id='predict', python_callable=run_predict, dag=dag)

t1 >> t2 >> t3 >> t4 >> t5
