import pandas as pd
import psycopg2


def fetch_data_from_db():
    try:
     
        # Database connection parameters
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="ecom",
            user="postgres",
            password="admin"
        )

        sales__table = 'sales'
        products__table = 'product'
        customers__table = 'customer'
        sales_df = pd.read_sql(f"SELECT * FROM {sales__table}", conn)
        product_df = pd.read_sql(f"SELECT * FROM {products__table}", conn)
        customer_df = pd.read_sql(f"SELECT * FROM {customers__table}", conn)
        print(sales_df.head())
        print('*'*20)
        print(product_df.head())
        print('*'*20)
        print(customer_df.head())

        conn.close()
        return {
            "sales": sales_df,
            "product": product_df,
            "customer": customer_df
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals():
            conn.close()