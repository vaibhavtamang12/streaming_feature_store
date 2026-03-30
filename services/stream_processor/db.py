import psycopg2
import os

def get_connection():
    return psycopg2.connect(
        host="postgres",
        database="feature_store",
        user="postgres",
        password="postgres"
    )


def insert_features(user_id, txn_count, txn_sum, timestamp):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO user_features (user_id, txn_count_1min, txn_sum_1min, event_timestamp)
        VALUES (%s, %s, %s, %s)
        """,
        (user_id, txn_count, txn_sum, timestamp)
    )

    conn.commit()
    cur.close()
    conn.close()