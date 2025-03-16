import React, { useState } from 'react';
import { metroStations, WHATSAPP_METRO_NUMBER } from '../data/metroYatriData';

interface MetroYatriProps {
  startLocation: {
    latitude: number;
    longitude: number;
    address: string;
  };
  finalDestination: {
    latitude: number;
    longitude: number;
    address: string;
  };
  onRideComplete: () => void;
}

export const MetroYatri: React.FC<MetroYatriProps> = ({
  startLocation,
  finalDestination,
  onRideComplete
}) => {
  const [rideStage, setRideStage] = useState<'first_ride' | 'metro' | 'last_ride'>('first_ride');
  
  // Find nearest metro stations to start and end points
  const nearestStartStation = metroStations[0]; // For demo, using first station
  const nearestEndStation = metroStations[1]; // For demo, using second station

  const handleWhatsAppBooking = () => {
    // Open WhatsApp with the metro number
    window.open(`https://wa.me/${WHATSAPP_METRO_NUMBER}?text=Hi`, '_blank');
  };

  const handleMetroRideComplete = () => {
    setRideStage('last_ride');
  };

  return (
    <div className="p-4 space-y-4">
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-xl font-bold mb-4">Metro Yatri Journey</h2>
        
        {rideStage === 'first_ride' && (
          <div>
            <p className="mb-2">First Ride: {startLocation.address} → {nearestStartStation.name}</p>
            <button
              onClick={() => setRideStage('metro')}
              className="bg-blue-500 text-white px-4 py-2 rounded"
            >
              Complete First Ride
            </button>
          </div>
        )}

        {rideStage === 'metro' && (
          <div>
            <p className="mb-2">Metro Journey: {nearestStartStation.name} → {nearestEndStation.name}</p>
            <button
              onClick={handleWhatsAppBooking}
              className="bg-green-500 text-white px-4 py-2 rounded mb-2 w-full"
            >
              Book Metro Ticket via WhatsApp
            </button>
            <button
              onClick={handleMetroRideComplete}
              className="bg-blue-500 text-white px-4 py-2 rounded w-full"
            >
              Metro Ride Complete
            </button>
          </div>
        )}

        {rideStage === 'last_ride' && (
          <div>
            <p className="mb-2">Final Ride: {nearestEndStation.name} → {finalDestination.address}</p>
            <button
              onClick={onRideComplete}
              className="bg-blue-500 text-white px-4 py-2 rounded"
            >
              Complete Journey
            </button>
          </div>
        )}

        <div className="mt-4 text-sm text-gray-600">
          <p>Journey Stages:</p>
          <ol className="list-decimal ml-4">
            <li className={rideStage === 'first_ride' ? 'font-bold' : ''}>
              Ride to {nearestStartStation.name} Metro Station
            </li>
            <li className={rideStage === 'metro' ? 'font-bold' : ''}>
              Metro Journey to {nearestEndStation.name}
            </li>
            <li className={rideStage === 'last_ride' ? 'font-bold' : ''}>
              Final ride to destination
            </li>
          </ol>
        </div>
      </div>
    </div>
  );
}; 