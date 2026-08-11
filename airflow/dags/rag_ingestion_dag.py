from __future__ import annotations

import logging
import os
import shlex
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

PROJECT_DIR = Path(
    os.getenv("RAG_PROJECT_DIR", Path(__file__).resolve().parents[2])
).resolve()
RAG_PYTHON = Path(os.getenv("RAG_PYTHON", PROJECT_DIR / ".venv" / "bin" / "python"))
SCHEDULE = os.getenv("RAG_AIRFLOW_SCHEDULE", "0 2 * * *")


def log_failure(context):
    logging.getLogger(__name__).error(
        "Échec du DAG %s, tâche %s, exécution %s",
        context["dag"].dag_id,
        context["task_instance"].task_id,
        context["run_id"],
    )


project = shlex.quote(str(PROJECT_DIR))
python = shlex.quote(str(RAG_PYTHON))

with DAG(
    dag_id="rag_ingestion_pipeline",
    description="Pipeline d'ingestion RAG",
    start_date=pendulum.datetime(2026, 8, 4, tz="Africa/Abidjan"),
    schedule=SCHEDULE,
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "moya",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
        "execution_timeout": timedelta(hours=1),
        "on_failure_callback": log_failure,
    },
    tags=["rag"],
) as dag:

    check_environment = BashOperator(
        task_id="check_environment",
        bash_command=(
            f"test -x {python} && test -f {project}/app.py "
            f"&& test -d {project}/data/documents"
        ),
    )

    run_pipeline = BashOperator(
        task_id="run_pipeline",
        bash_command=f"cd {project} && {python} app.py",
    )

    validate_index = BashOperator(
        task_id="validate_index",
        bash_command=f"cd {project} && {python} validate.py",
    )

    check_environment >> run_pipeline >> validate_index
    
