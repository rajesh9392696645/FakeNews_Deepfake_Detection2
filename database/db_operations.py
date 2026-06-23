import sqlite3
import os

DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "prediction_history.db"
)

# ------------------------------------------
# Database Connection
# ------------------------------------------

def get_connection():
    return sqlite3.connect(DB_PATH)

# ------------------------------------------
# User Operations
# ------------------------------------------

def register_user(username, email, password):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute("""
        INSERT INTO users
        (username,email,password)
        VALUES (?,?,?)
        """,
        (username,email,password))

        conn.commit()

        return True

    except sqlite3.IntegrityError:

        return False

    finally:

        conn.close()


def login_user(email,password):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM users
    WHERE email=?
    AND password=?
    """,
    (email,password))

    user = cursor.fetchone()

    conn.close()

    return user

# ------------------------------------------
# Prediction Operations
# ------------------------------------------

def save_prediction(
        username,
        file_name,
        prediction,
        confidence,
        detection_type):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO prediction_history
    (
        username,
        file_name,
        prediction,
        confidence,
        detection_type
    )
    VALUES
    (?, ?, ?, ?, ?)
    """,
    (
        username,
        file_name,
        prediction,
        confidence,
        detection_type
    ))

    conn.commit()
    conn.close()


def get_all_predictions():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM prediction_history
    ORDER BY created_at DESC
    """)

    data = cursor.fetchall()

    conn.close()

    return data


def get_user_predictions(username):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM prediction_history
    WHERE username=?
    ORDER BY created_at DESC
    """,
    (username,))

    data = cursor.fetchall()

    conn.close()

    return data


def delete_prediction(record_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    DELETE FROM prediction_history
    WHERE id=?
    """,
    (record_id,))

    conn.commit()
    conn.close()

# ------------------------------------------
# Dashboard Statistics
# ------------------------------------------

def get_total_predictions():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT COUNT(*)
    FROM prediction_history
    """)

    count = cursor.fetchone()[0]

    conn.close()

    return count


def get_fake_count():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT COUNT(*)
    FROM prediction_history
    WHERE prediction='Fake'
    """)

    count = cursor.fetchone()[0]

    conn.close()

    return count


def get_real_count():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT COUNT(*)
    FROM prediction_history
    WHERE prediction='Real'
    """)

    count = cursor.fetchone()[0]

    conn.close()

    return count