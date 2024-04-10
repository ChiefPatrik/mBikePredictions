import requests
import random
import datetime

def generate_random_date():
    start_date = datetime.datetime(2023, 1, 1)
    end_date = datetime.datetime(2023, 12, 31)
    delta = end_date - start_date
    random_date = start_date + datetime.timedelta(days=random.randint(0, delta.days))
    return random_date.strftime("%Y-%m-%d %H:%M:%S+00:00")

def generate_random_data_window(size):
    data = []
    for _ in range(size):
        data.append({
            "date": generate_random_date(),
            "available_bike_stands": random.randint(0, 20)
        })
    return {"data": data}

def test_predict_mBike():
    url = "http://127.0.0.1:3000/mbajk/predict/"
    data_window = generate_random_data_window(186)
    try:
        response = requests.post(url, json=data_window)
        assert response.status_code == 200
        print("Test passed: Status code 200 received.")
    except AssertionError:
        print("Test failed: Unexpected status code received:", response.status_code)
    except Exception as e:
        print("Test failed with exception:", e)

if __name__ == "__main__":
    test_predict_mBike()
