import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [stationsData, setStationsData] = useState(Array(29).fill(null));
  const [clickedStation, setClickedStation] = useState(null);
  const [predictions, setPredictions] = useState(Array(7).fill(null));
  const [showPredictions, setShowPredictions] = useState(false);
  const [selectedStationNumber, setSelectedStationNumber] = useState(null);

  const contract = "maribor";
  const api_key = "5e150537116dbc1786ce5bec6975a8603286526b";
  const stationsUrl = `https://api.jcdecaux.com/vls/v1/stations?contract=${contract}&apiKey=${api_key}`;

  const fetchStationData = async (stationNumber) => {
    try {
      const response = await axios.get(stationsUrl);
      setStationsData(response.data);
      setClickedStation(stationNumber);
    } catch (error) {
      console.error('Error fetching station data:', error);
    }
  };

  const handleStationClick = async (stationNumber) => {
    await fetchStationData(stationNumber);
    setClickedStation(stationNumber);
    setSelectedStationNumber(stationNumber);
    // setShowPredictions(false);
  };

  const handlePredictClick = async (stationNumber) => {
    try {
      console.log('stationNumber:', stationNumber);
      const response = await axios.post('http://localhost:3000/mbajk/predict/', {
        data: [{ station_number: stationNumber }]
      });
      console.log('response:', response.data.predictions)
      setPredictions(response.data.predictions);
      setShowPredictions(true);
      setSelectedStationNumber(stationNumber);
    } catch (error) {
      console.error('Error fetching predictions:', error);
    }
  };

  const renderPredictions = () => {
    const currentTime = new Date();
    return Array.from({ length: 7 }, (_, index) => {
      const time = new Date(currentTime.getTime() + (index + 1) * 3600000);
      return (
        <div key={index} className="prediction-row">
          <div className="time">{`${time.getHours()}:${time.getMinutes().toString().padStart(2, '0')}`}</div>
          <div className="prediction">{predictions[index]}</div>
        </div>
      );
    });
  };
  

  return (
    <div className="app-wrapper">
      <div className="app-container">
        {stationsData.map((station, index) => (
          <div key={index + 1} onClick={() => handleStationClick(index + 1)} className={`station-card${clickedStation === index + 1 ? ' active' : ''}`}>
            <h2>Station {index + 1}</h2>
            {clickedStation === index + 1 && (
              <>
                <div>
                  <p>Name: {station.name}</p>
                  <p>Address: {station.address}</p>
                  <p>Bike stands: {station.bike_stands}</p>
                  <div>
                    <img src="/bike.png" alt="Available Bikes" className="icon" /><span style={{ fontSize: '16px', marginRight: '20px' }}>{station.available_bikes}</span>
                    <img src="/parking.png" alt="Available Bike Stands" className="icon" /><span style={{ fontSize: '16px', marginRight: '20px' }}>{station.available_bike_stands}</span>
                  </div>
                  <button onClick={() => handlePredictClick(station.number)}>Predict available bike stands</button>
                </div>
                {showPredictions && selectedStationNumber === station.number && (
                  <div className="predictions-container">
                    {renderPredictions()}
                  </div>
                )}
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
