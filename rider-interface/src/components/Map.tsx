import React, { useMemo } from "react";
import { GoogleMap, useLoadScript, HeatmapLayer, Marker } from "@react-google-maps/api";
import { driverDensityData, trafficData } from "@/data/newdummydata";

interface Location {
  latitude: number;
  longitude: number;
  address: string;
}

interface MapProps {
  pickup: Location;
  destination: Location;
  driverLocation?: { latitude: number; longitude: number } | null;
  className?: string;
  showHeatmap?: boolean;
  drivers?: any[];
  waypoints?: Location[];
}

const getIconForVehicle = (vehicleType: string) => {
  if (vehicleType === "bike") return "/bike-icon.png";
  if (vehicleType === "auto") return "/auto-icon.png";
  if (vehicleType === "cab") return "/cab-icon.png";
  if (vehicleType === "carpool") return "/carpool-icon.png";
  return "";
};

// Function to generate a small random offset (approx ±0.0005 degrees)
const randomOffset = () => (Math.random() - 0.5) * 0.001;

const Map: React.FC<MapProps> = ({
  pickup,
  destination,
  driverLocation,
  className = "",
  showHeatmap = false,
  drivers = [],
  waypoints = []
}) => {
  const { isLoaded, loadError } = useLoadScript({
    googleMapsApiKey: "AIzaSyAtxaaw2GuXURkOtr6cq0Kzd_Tw-e8Xogw", // Replace with your actual API key.
    libraries: ["visualization"],
  });

  const center = useMemo(
    () => ({
      lat: pickup.latitude,
      lng: pickup.longitude,
    }),
    [pickup]
  );

  if (loadError) return <div>Error loading maps</div>;
  if (!isLoaded) return <div>Loading Maps...</div>;

  return (
    <div className={`bg-gray-100 rounded-lg p-4 ${className}`}>
      <div className="text-center text-gray-500">
        <p>Map View</p>
        <p className="text-sm">From: {pickup.address}</p>
        {waypoints.map((point, index) => (
          <p key={index} className="text-sm">Via: {point.address}</p>
        ))}
        <p className="text-sm">To: {destination.address}</p>
      </div>

      <GoogleMap mapContainerStyle={{ width: "100%", height: "400px" }} center={center} zoom={12}>
        {/* Randomize pickup and destination markers */}
        <Marker
          position={{ lat: pickup.latitude + randomOffset(), lng: pickup.longitude + randomOffset() }}
          label="P"
        />
        <Marker
          position={{ lat: destination.latitude + randomOffset(), lng: destination.longitude + randomOffset() }}
          label="D"
        />
        {driverLocation && (
          <Marker
            position={{ lat: driverLocation.latitude, lng: driverLocation.longitude }}
            label="Driver"
          />
        )}
        {drivers && drivers.map(driver => (
          <Marker
            key={driver.id}
            position={{ lat: driver.latitude, lng: driver.longitude }}
            icon={{
              url: getIconForVehicle(driver.vehicleType),
              scaledSize: new google.maps.Size(24, 24)
            }}
          />
        ))}
        {showHeatmap && (
          <>
            <HeatmapLayer
              data={driverDensityData.map(
                (point) => new google.maps.LatLng(point.lat, point.lng)
              )}
              options={{
                radius: 20,
                gradient: ["rgba(0,0,255,0)", "rgba(0,0,255,1)"],
              }}
            />
            <HeatmapLayer
              data={trafficData.map(
                (point) => new google.maps.LatLng(point.lat, point.lng)
              )}
              options={{
                radius: 20,
                gradient: ["rgba(255,0,0,0)", "rgba(255,0,0,1)"],
              }}
            />
          </>
        )}
      </GoogleMap>
    </div>
  );
};

export default Map;
