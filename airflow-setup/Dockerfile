FROM apache/airflow:3.1.5-python3.13

USER airflow

COPY requirements-project.txt /tmp/requirements-project.txt

RUN pip install --no-cache-dir \
    "apache-airflow==3.1.5" \
    "apache-airflow-providers-fab" \
    -r /tmp/requirements-project.txt
