import requests

def test_API():
    contract = "maribor" 
    api_key = "5e150537116dbc1786ce5bec6975a8603286526b"  
    url = f"https://api.jcdecaux.com/vls/v1/stations?contract={contract}&apiKey={api_key}"
    try:
        response = requests.get(url)
        assert response.status_code == 200
        print("Test passed: Status code 200 received - Station API is active.")
    except AssertionError:
        print("Test failed: Unexpected response format or status code.")
    except Exception as e:
        print("Test failed with exception:", e)

if __name__ == "__main__":
    test_API()