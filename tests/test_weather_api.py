import requests

def test_API():
    url = "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&hourly=temperature_2m"
    try:
        response = requests.get(url)
        assert response.status_code == 200
        print("Test passed: Status code 200 received - Weather API is active.")
    except AssertionError:
        print("Test failed: Unexpected response format or status code.")
    except Exception as e:
        print("Test failed with exception:", e)

if __name__ == "__main__":
    test_API()