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
      const sortedData = response.data.sort((a, b) => a.number - b.number);   // Sort the response data by the 'number' field
      setStationsData(sortedData);
      setClickedStation(stationNumber);
    } catch (error) {
      console.error('Error fetching station data:', error);
    }
  };

  const handleStationClick = async (stationNumber) => {
    await fetchStationData(stationNumber);
    setClickedStation(stationNumber);
    setSelectedStationNumber(stationNumber);
    setShowPredictions(false);
  };

  const handlePredictClick = async (stationNumber) => {
    try {
      console.log('stationNumber:', stationNumber);
      const response = await axios.post(`${process.env.REACT_APP_PREDICTION_API_URL}/mbajk/predict/`, {
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
    const nextFullHour = new Date(currentTime);
    nextFullHour.setHours(nextFullHour.getHours() + 1);
    nextFullHour.setMinutes(0);
    nextFullHour.setSeconds(0);
  
    return (
      <div className="predictions-container">
        {Array.from({ length: 7 }, (_, index) => {
          const time = new Date(nextFullHour.getTime() + index * 3600000);
          return (
            <div key={index} className="prediction-row">
              <div className="time">{`${time.getHours()}:${time.getMinutes().toString().padStart(2, '0')}`}</div>
              <img src="/parking.png" alt="Available Bike Stands" className="icon" />
              <div className="prediction">{predictions[index]}</div>
            </div>
          );
        })}
      </div>
    );
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
                  <p>NAME: {station.name}</p>
                  <p>ADDRESS: {station.address}</p>
                  <p>BIKE STANDS: {station.bike_stands}</p>
                  <div className="info-container">
                    <img src="/bike.png" alt="Available Bikes" className="icon" /><span style={{ fontSize: '30px', marginRight: '20px' }}>{station.available_bikes}</span>
                    <img src="/parking.png" alt="Available Bike Stands" className="icon" /><span style={{ fontSize: '30px', marginRight: '20px' }}>{station.available_bike_stands}</span>
                  </div>
                  <div className="info-container">
                    <button className="predict-button" onClick={() => handlePredictClick(station.number)}>PREDICT</button>
                  </div>
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
