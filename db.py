import sqlite3
import os

DATABASE_PATH = os.path.join(
    "database",
    "prediction_history.db"
)

def get_db_connection():

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.row_factory = sqlite3.Row

    return conn


def execute_query(query,
                  params=()):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        query,
        params
    )

    conn.commit()

    conn.close()


def fetch_all(query,
              params=()):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        query,
        params
    )

    data = cursor.fetchall()

    conn.close()

    return data


def fetch_one(query,
              params=()):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        query,
        params
    )

    data = cursor.fetchone()

    conn.close()

    return data