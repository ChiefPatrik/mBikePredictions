import requests
import random
import datetime

def generate_random_station_number():
    return {"data": [{"station_number": random.randint(1, 29)}]}

def test_predict_mBike():
    url = "http://127.0.0.1:3000/mbajk/predict/"
    station_number = generate_random_station_number()
    try:
        response = requests.post(url, json=station_number)
        assert response.status_code == 200
        predictions = response.json().get("predictions")
        assert isinstance(predictions, list) and len(predictions) == 7
        print("Test passed: Status code 200 received, and predictions received.")
    except AssertionError:
        print("Test failed: Unexpected response format or status code.")
    except Exception as e:
        print("Test failed with exception:", e)

if __name__ == "__main__":
    test_predict_mBike()