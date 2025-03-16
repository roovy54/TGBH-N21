import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { calculateDistance, calculateETA, calculateFare, formatTime } from "@/utils/rideUtils";
import FareBreakdown from "./FareBreakdown";

interface VehicleOption {
  id: string;
  name: string;
  image: string;
}

interface PreScheduleRideProps {
  vehicleOptions: VehicleOption[];
  onSchedule: (
    scheduledTime: Date,
    vehicleType: string,
    pickup: string,
    destination: string,
    finalFare: number
  ) => void;
  onCancel: () => void;
}

const PreScheduleRide: React.FC<PreScheduleRideProps> = ({ vehicleOptions, onSchedule, onCancel }) => {
  const [prePickup, setPrePickup] = useState<string>("");
  const [preDestination, setPreDestination] = useState<string>("");
  const [scheduledTime, setScheduledTime] = useState<string>("");
  const [selectedVehicle, setSelectedVehicle] = useState<string>("");
  const [computedRideOption, setComputedRideOption] = useState<any>(null);
  const [computedFare, setComputedFare] = useState<number>(0);
  const [showFareBreakdown, setShowFareBreakdown] = useState<boolean>(false);

  const handleComputeFare = () => {
    if (!prePickup || !preDestination || !scheduledTime || !selectedVehicle) {
      alert("Please fill all fields and select a vehicle type.");
      return;
    }
    // For demonstration purposes, we'll simulate coordinates.
    // In a production app, you would convert addresses into coordinates via a geocoding API.
    const userLocation = { latitude: 12.9716, longitude: 77.5946 };
    const pickupCoords = userLocation; // using a fixed point for demo
    const destinationCoords = {
      latitude: userLocation.latitude + 0.02,
      longitude: userLocation.longitude + 0.02,
    };

    const distance = calculateDistance(
      pickupCoords.latitude,
      pickupCoords.longitude,
      destinationCoords.latitude,
      destinationCoords.longitude
    );
    const eta = calculateETA(distance, selectedVehicle);
    const fare = calculateFare(distance, eta, selectedVehicle);
    const formattedTime = formatTime(eta);

    // Find selected vehicle object
    const vehicleObj = vehicleOptions.find((v) => v.id === selectedVehicle);
    const rideOption = {
      id: selectedVehicle,
      name: vehicleObj ? vehicleObj.name : "",
      image: vehicleObj ? vehicleObj.image : "",
      estimated_time: formattedTime,
      price: fare,
      ...(selectedVehicle === "bike" ? { promocode: "FIRST10", discount: 10 } : {}),
    };

    setComputedFare(fare);
    setComputedRideOption(rideOption);
    setShowFareBreakdown(true);
  };

  const handleAcceptFare = (finalFare: number) => {
    onSchedule(new Date(scheduledTime), selectedVehicle, prePickup, preDestination, finalFare);
  };

  return (
    <div className="bg-white p-6 shadow-lg rounded-lg max-w-md mx-auto space-y-4">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Pre-schedule a Ride</h2>
      {!showFareBreakdown && (
        <>
          <div>
            <label className="block text-gray-700 font-medium mb-1">Pickup Location:</label>
            <input
              type="text"
              value={prePickup}
              onChange={(e) => setPrePickup(e.target.value)}
              placeholder="Enter pickup location"
              className="w-full p-2 border rounded"
            />
          </div>
          <div>
            <label className="block text-gray-700 font-medium mb-1">Destination:</label>
            <input
              type="text"
              value={preDestination}
              onChange={(e) => setPreDestination(e.target.value)}
              placeholder="Enter destination"
              className="w-full p-2 border rounded"
            />
          </div>
          <div>
            <label className="block text-gray-700 font-medium mb-1">Select Date & Time:</label>
            <input
              type="datetime-local"
              value={scheduledTime}
              onChange={(e) => setScheduledTime(e.target.value)}
              className="w-full p-2 border rounded"
            />
          </div>
          <div>
            <label className="block text-gray-700 font-medium mb-1">Select Vehicle Type:</label>
            <div className="flex gap-4">
              {vehicleOptions.map((option) => (
                <Button
                  key={option.id}
                  variant={selectedVehicle === option.id ? "default" : "outline"}
                  onClick={() => setSelectedVehicle(option.id)}
                  className="py-1 px-3 text-sm"
                >
                  {option.name}
                </Button>
              ))}
            </div>
          </div>
          <div className="flex justify-center gap-6 mt-4">
            <Button
              onClick={handleComputeFare}
              className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-6 rounded-lg"
            >
              Compute Fare
            </Button>
            <Button
              onClick={onCancel}
              className="bg-gray-500 hover:bg-gray-600 text-white py-2 px-6 rounded-lg"
            >
              Cancel
            </Button>
          </div>
        </>
      )}
      {showFareBreakdown && computedRideOption && (
        <FareBreakdown
          baseFare={computedRideOption.price}
          rideOption={computedRideOption}
          onAccept={handleAcceptFare}
          onDeny={() => setShowFareBreakdown(false)}
        />
      )}
    </div>
  );
};

export default PreScheduleRide;