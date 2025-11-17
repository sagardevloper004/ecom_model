from getData import fetch_data_from_db

if __name__ == "__main__":
    data = fetch_data_from_db()
    # You can now use the fetched data as needed
    print(data)