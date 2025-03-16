import React, { useState, useEffect } from "react";
import LocationSearch from "./LocationSearch";
import RideOptions from "./RideOptions";
import FareBreakdown from "./FareBreakdown";
import DriverMatch from "./DriverMatch";
import RideTracker from "./RideTracker";
import Map from "./Map";
import PreScheduleRide from "./PreScheduleRide";
import { MetroYatri } from "./MetroYatri";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { 
  calculateDistance, 
  calculateETA, 
  calculateFare, 
  findNearestDrivers, 
  batchedMatchDrivers,
  formatTime 
} from "@/utils/rideUtils";
import { aiMatchScore } from "@/utils/aiMatch";
import { 
  userLocation, 
  popularDestinations, 
  vehicleTypes, 
  availableDrivers 
} from "@/data/dummyData";
import { metroStations, WHATSAPP_METRO_NUMBER } from "@/data/metroYatriData";
import { cn } from "@/lib/utils";

// Types
interface RideOption {
  id: string;
  name: string;
  image: string;
  estimated_time: string;
  price: number;
  promocode?: string;
  discount?: number;
}

interface Driver {
  id: string;
  name: string;
  photo: string;
  rating: number;
  vehicleNumber: string;
  distance: number;
  eta: number;
}

interface Location {
  latitude: number;
  longitude: number;
  address: string;
}

interface MetroJourneyDetails {
  startStation: typeof metroStations[0] | null;
  endStation: typeof metroStations[0] | null;
  firstRideDriver: Driver | null;
  lastRideDriver: Driver | null;
  firstLegFare: number;
  lastLegFare: number;
  metroFare: number;
  firstLegTime: number;
  lastLegTime: number;
  metroTime: number;
}

// Added new steps "fareBreakdown", "schedule", "scheduled", and Metro Yatri steps to the union
type BookingStep = "location" | "options" | "fareBreakdown" | "matching" | "tracking" | "schedule" | "scheduled" | "metroyatri_location" | "metroyatri_first_ride" | "metroyatri_metro" | "metroyatri_last_ride" | "metroyatri_summary" | "immediate_location";

const RideBooking: React.FC = () => {
  const [step, setStep] = useState<BookingStep>("location");
  const [pickup, setPickup] = useState<Location>(userLocation);
  const [destination, setDestination] = useState<Location | null>(null);
  const [distanceKm, setDistanceKm] = useState<number>(0);
  const [selectedVehicleType, setSelectedVehicleType] = useState<string | "metroyatri">("");
  const [selectedRide, setSelectedRide] = useState<RideOption | null>(null);
  const [selectedDriver, setSelectedDriver] = useState<Driver | null>(null);
  const [availableRideOptions, setAvailableRideOptions] = useState<RideOption[]>([]);
  const [nearestDrivers, setNearestDrivers] = useState<Driver[]>([]);
  const [driverLocation, setDriverLocation] = useState<{latitude: number, longitude: number} | null>(null);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);
  const [womenSafetyMode, setWomenSafetyMode] = useState<boolean>(false);
  // Now storing an array of scheduled rides
  const [scheduledRides, setScheduledRides] = useState<{ time: Date; vehicleType: string; pickup: string; destination: string; fare: number }[]>([]);
  const [metroJourney, setMetroJourney] = useState<MetroJourneyDetails>({
    startStation: null,
    endStation: null,
    firstRideDriver: null,
    lastRideDriver: null,
    firstLegFare: 0,
    lastLegFare: 0,
    metroFare: 0,
    firstLegTime: 0,
    lastLegTime: 0,
    metroTime: 0
  });

  // Calculate options when destination changes
  useEffect(() => {
    if (destination) {
      const distance = calculateDistance(
        pickup.latitude, 
        pickup.longitude, 
        destination.latitude, 
        destination.longitude
      );
      setDistanceKm(distance);
      
      const options = vehicleTypes.map(vehicle => {
        const etaMinutes = calculateETA(distance, vehicle.id);
        let fare = calculateFare(distance, etaMinutes, vehicle.id);
        
        // Optionally adjust pricing for carpool – e.g. 20% discount
        if (vehicle.id === "carpool") {
          fare = fare * 0.8;
        }
        
        return {
          id: vehicle.id,
          name: vehicle.name,
          image: vehicle.image,
          estimated_time: formatTime(etaMinutes),
          price: fare,
          ...(vehicle.id === "bike" ? { promocode: "FIRST10", discount: 10 } : {})
        };
      });
      
      setAvailableRideOptions(options);
    }
  }, [destination, pickup]);

  const calculateMetroJourneyDetails = (pickup: Location, destination: Location, startStation: typeof metroStations[0], endStation: typeof metroStations[0]) => {
    // Calculate first leg details
    const firstLegDistance = calculateDistance(
      pickup.latitude,
      pickup.longitude,
      startStation.latitude,
      startStation.longitude
    );
    const firstLegTime = calculateETA(firstLegDistance, "auto");
    const firstLegFare = calculateFare(firstLegDistance, firstLegTime, "auto");

    // Calculate metro journey details
    const metroDistance = calculateDistance(
      startStation.latitude,
      startStation.longitude,
      endStation.latitude,
      endStation.longitude
    );
    const metroTime = Math.ceil(metroDistance * 3); // Assuming 3 minutes per km for metro
    const metroFare = Math.ceil(metroDistance * 10); // Assuming ₹10 per km for metro

    // Calculate last leg details
    const lastLegDistance = calculateDistance(
      endStation.latitude,
      endStation.longitude,
      destination.latitude,
      destination.longitude
    );
    const lastLegTime = calculateETA(lastLegDistance, "auto");
    const lastLegFare = calculateFare(lastLegDistance, lastLegTime, "auto");

    return {
      firstLegTime,
      firstLegFare,
      metroTime,
      metroFare,
      lastLegTime,
      lastLegFare
    };
  };

  const handleLocationsSelected = (pickupLoc: string, destinationLoc: string) => {
    const foundPickup = popularDestinations.find(
      dest => dest.address.toLowerCase().includes(pickupLoc.toLowerCase()) ||
              pickupLoc.toLowerCase().includes(dest.name.toLowerCase())
    );
    const foundDestination = popularDestinations.find(
      dest => dest.address.toLowerCase().includes(destinationLoc.toLowerCase()) ||
              destinationLoc.toLowerCase().includes(dest.name.toLowerCase())
    );
    
    if (foundPickup) {
      setPickup(foundPickup);
    } else {
      setPickup({
        latitude: userLocation.latitude,
        longitude: userLocation.longitude,
        address: pickupLoc
      });
    }
    
    if (foundDestination) {
      setDestination(foundDestination);
    } else {
      setDestination({
        latitude: userLocation.latitude + 0.02,
        longitude: userLocation.longitude + 0.02,
        address: destinationLoc
      });
    }
    
    if (step === "metroyatri_location") {
      // Find nearest metro stations
      const nearestStartStation = metroStations[0];
      const nearestEndStation = metroStations[1];
      
      const journeyDetails = calculateMetroJourneyDetails(
        pickup,
        destination as Location,
        nearestStartStation,
        nearestEndStation
      );

      setMetroJourney(prev => ({
        ...prev,
        startStation: nearestStartStation,
        endStation: nearestEndStation,
        ...journeyDetails
      }));

      // Find drivers for first leg
      const drivers = batchedMatchDrivers(
        pickup.latitude,
        pickup.longitude,
        availableDrivers,
        "auto",
        3,
        2
      );
      
      const mappedDrivers = drivers.map(driver => ({
        id: driver.id,
        name: womenSafetyMode ? "Shruti Sharma" : driver.name,
        photo: driver.photo,
        rating: driver.rating,
        vehicleNumber: driver.vehicleNumber,
        distance: driver.distance,
        eta: driver.eta
      }));
      
      setNearestDrivers(mappedDrivers);
      setStep("metroyatri_first_ride");
    } else {
      setStep("options");
    }
  };

  // When a ride option is selected, show fare breakdown instead of going directly to matching.
  const handleRideSelected = (option: RideOption) => {
    setSelectedRide(option);
    setSelectedVehicleType(option.id);
    if (option.id === "metroyatri") {
      setStep("metroyatri_location");
    } else {
      setStep("fareBreakdown");
    }
  };

  // Called when the user accepts the breakdown with an added extra amount.
  const handleFareAccepted = async (finalFare: number) => {
    if (destination && selectedRide) {
      const updatedRide = { ...selectedRide, price: finalFare };
      setSelectedRide(updatedRide);
      
      const drivers = batchedMatchDrivers(
        pickup.latitude,
        pickup.longitude,
        availableDrivers,
        selectedRide.id,
        3,
        2
      );
      
      const driversWithAIScore = await Promise.all(
        drivers.map(async (driver) => {
          const score = await aiMatchScore({ pickup, destination, rideOption: selectedRide }, driver);
          return { ...driver, aiScore: score };
        })
      );
      
      driversWithAIScore.sort((a, b) => b.aiScore - a.aiScore);
      
      const mappedDrivers = driversWithAIScore.map(driver => ({
        id: driver.id,
        name: womenSafetyMode ? "Shruti Sharma" : driver.name,
        photo: driver.photo,
        rating: driver.rating,
        vehicleNumber: driver.vehicleNumber,
        distance: driver.distance,
        eta: driver.eta
      }));
      
      setNearestDrivers(mappedDrivers);
      setStep("matching");
    }
  };

  const handleFareDenied = () => {
    // Return back to the ride options screen
    setStep("options");
  };

  const handleDriverConfirmed = (driver: Driver) => {
    setSelectedDriver(driver);
    const driverData = availableDrivers.find(d => d.id === driver.id);
    if (driverData) {
      setDriverLocation({
        latitude: driverData.latitude,
        longitude: driverData.longitude
      });
    }
    
    setStep("tracking");
  };

  const handleGoBack = () => {
    switch (step) {
      case "immediate_location":
      case "metroyatri_location":
        setStep("location");
        break;
      case "options":
        setStep("immediate_location");
        break;
      case "fareBreakdown":
        setStep("options");
        break;
      case "matching":
        setStep("options");
        break;
      case "tracking":
        setStep("matching");
        break;
      case "scheduled":
        setStep("location");
        break;
      default:
        setStep("location");
    }
  };

  const handleRideComplete = () => {
    setTimeout(() => {
      setStep("location");
      setDestination(null);
      setSelectedRide(null);
      setSelectedDriver(null);
      setDriverLocation(null);
    }, 2000);
  };

  // When a ride is pre-scheduled, add it to the list and navigate to the Scheduled screen.
  const handlePreSchedule = (
    scheduledTime: Date,
    vehicleType: string,
    pickupLocation: string,
    destinationLocation: string,
    finalFare: number
  ) => {
    const newRide = {
      time: scheduledTime,
      vehicleType,
      pickup: pickupLocation,
      destination: destinationLocation,
      fare: finalFare,
    };
    setScheduledRides((prev) => [...prev, newRide]);
    setStep("scheduled");
  };

  // Handle first leg driver confirmation
  const handleFirstLegDriverConfirmed = async (driver: Driver) => {
    setMetroJourney(prev => ({
      ...prev,
      firstRideDriver: driver
    }));
    setStep("metroyatri_metro");
  };

  // Handle last leg driver confirmation
  const handleLastLegDriverConfirmed = async (driver: Driver) => {
    setMetroJourney(prev => ({
      ...prev,
      lastRideDriver: driver
    }));
    setStep("metroyatri_summary");
  };

  // Handle metro ride completion
  const handleMetroRideComplete = () => {
    // Find drivers for last leg
    if (metroJourney.endStation && destination) {
      const drivers = batchedMatchDrivers(
        metroJourney.endStation.latitude,
        metroJourney.endStation.longitude,
        availableDrivers,
        "auto", // Default to auto for metro rides
        3,
        2
      );
      
      const mappedDrivers = drivers.map(driver => ({
        id: driver.id,
        name: driver.name,
        photo: driver.photo,
        rating: driver.rating,
        vehicleNumber: driver.vehicleNumber,
        distance: driver.distance,
        eta: driver.eta
      }));
      
      setNearestDrivers(mappedDrivers);
      setStep("metroyatri_last_ride");
    }
  };

  // Handle journey completion from summary
  const handleJourneyComplete = () => {
    setTimeout(() => {
      setStep("location");
      setDestination(null);
      setSelectedRide(null);
      setSelectedDriver(null);
      setDriverLocation(null);
      setMetroJourney({
        startStation: null,
        endStation: null,
        firstRideDriver: null,
        lastRideDriver: null,
        firstLegFare: 0,
        lastLegFare: 0,
        metroFare: 0,
        firstLegTime: 0,
        lastLegTime: 0,
        metroTime: 0
      });
    }, 2000);
  };

  return (
    <div className="w-full max-w-md mx-auto">
      {step !== "location" && (
        <div className="flex items-center justify-between mb-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={handleGoBack}
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div className="flex items-center gap-2">
            <label className="text-sm text-muted-foreground">Women's Safety</label>
            <Button
              variant={womenSafetyMode ? "default" : "outline"}
              size="sm"
              onClick={() => setWomenSafetyMode(!womenSafetyMode)}
              className={cn(
                "transition-colors",
                womenSafetyMode && "bg-pink-600 hover:bg-pink-700"
              )}
            >
              {womenSafetyMode ? "On" : "Off"}
            </Button>
          </div>
        </div>
      )}
      
      {step !== "location" && destination && (
        <Button variant="outline" onClick={() => setShowHeatmap(prev => !prev)} className="mb-4">
          { showHeatmap ? "Hide Heatmap" : "Show Heatmap" }
        </Button>
      )}
      
      {step !== "location" && destination && (
        <Map 
          pickup={pickup}
          destination={destination}
          driverLocation={driverLocation}
          className="mb-4"
          showHeatmap={showHeatmap}
          drivers={availableDrivers}
        />
      )}
      
      {step === "location" && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 gap-4">
            <Button 
              variant="outline" 
              className="w-full h-16 text-lg font-semibold"
              onClick={() => setStep("scheduled")}
            >
              Scheduled Rides
            </Button>
            <Button 
              variant="outline"
              className="w-full h-16 text-lg font-semibold"
              onClick={() => setStep("immediate_location")}
            >
              Immediate Ride
            </Button>
            <Button 
              variant="outline"
              className="w-full h-16 text-lg font-semibold"
              onClick={() => setStep("schedule")}
            >
              Pre-schedule Ride
            </Button>
            <Button 
              variant="outline"
              className="w-full h-16 text-lg font-semibold"
              onClick={() => {
                setSelectedVehicleType("metroyatri");
                setStep("metroyatri_location");
              }}
            >
              Metro Yatri
            </Button>
          </div>
        </div>
      )}

      {/* Location search for immediate rides */}
      {step === "immediate_location" && (
        <div>
          <h2 className="text-xl font-bold mb-4">Book an Immediate Ride</h2>
          <LocationSearch onLocationsSelected={handleLocationsSelected} />
        </div>
      )}

      {/* Location search for metro yatri with special UI */}
      {step === "metroyatri_location" && (
        <div>
          <div className="mb-4 p-4 bg-blue-50 rounded-lg">
            <h2 className="text-lg font-semibold mb-2">Metro Yatri Journey</h2>
            <p className="text-sm text-gray-600">Enter your pickup and destination locations to find the best metro route.</p>
          </div>
          <LocationSearch onLocationsSelected={handleLocationsSelected} />
        </div>
      )}
      
      {step === "options" && (
        <RideOptions 
          options={availableRideOptions} 
          onSelect={handleRideSelected}
          distance={distanceKm}
        />
      )}
      
      {step === "fareBreakdown" && selectedRide && (
        <FareBreakdown 
          baseFare={selectedRide.price} 
          rideOption={selectedRide} 
          onAccept={handleFareAccepted}
          onDeny={handleFareDenied}
        />
      )}
      
      {step === "matching" && (
        <DriverMatch 
          drivers={nearestDrivers}
          onDriverConfirmed={handleDriverConfirmed}
          onCancel={() => setStep("options")}
        />
      )}
      
      {step === "tracking" && selectedRide && selectedDriver && destination && (
        <RideTracker 
          driver={selectedDriver}
          pickup={pickup.address}
          destination={destination.address}
          fare={selectedRide.price}
          estimatedTime={selectedRide.estimated_time}
          distance={distanceKm}
          onRideComplete={handleRideComplete}
        />
      )}

      {step === "schedule" && (
        <PreScheduleRide
          vehicleOptions={vehicleTypes}
          onSchedule={handlePreSchedule}
          onCancel={() => setStep("location")}
        />
      )}

      {step === "scheduled" && (
        <div>
          <h2 className="text-2xl font-bold mb-4">Scheduled Rides</h2>
          {scheduledRides.length === 0 ? (
            <p>No scheduled rides yet.</p>
          ) : (
            <ul className="space-y-3">
              {scheduledRides.map((ride, index) => (
                <li key={index} className="border p-3 rounded">
                  <p className="font-medium">Vehicle Type: {ride.vehicleType}</p>
                  <p>Pickup: {ride.pickup}</p>
                  <p>Destination: {ride.destination}</p>
                  <p>Fare: ₹{ride.fare.toFixed(2)}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {step === "metroyatri_first_ride" && metroJourney.startStation && (
        <div>
          <h2 className="text-xl font-bold mb-4">First Leg: To Metro Station</h2>
          <div className="mb-4 p-4 bg-blue-50 rounded-lg">
            <p className="mb-2">Ride to {metroJourney.startStation.name} Metro Station</p>
            <p className="text-sm text-gray-600">Estimated Time: {formatTime(metroJourney.firstLegTime)}</p>
            <p className="text-sm text-gray-600">Auto Fare: ₹{metroJourney.firstLegFare.toFixed(2)}</p>
          </div>
          <Map 
            pickup={pickup}
            destination={{
              latitude: metroJourney.startStation.latitude,
              longitude: metroJourney.startStation.longitude,
              address: metroJourney.startStation.name
            }}
            driverLocation={driverLocation}
            className="mb-4"
            showHeatmap={false}
            drivers={availableDrivers}
          />
          <DriverMatch 
            drivers={nearestDrivers}
            onDriverConfirmed={handleFirstLegDriverConfirmed}
            onCancel={() => setStep("location")}
          />
        </div>
      )}

      {step === "metroyatri_metro" && metroJourney.startStation && metroJourney.endStation && (
        <div className="p-4 space-y-4">
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="text-xl font-bold mb-4">Metro Journey</h2>
            <div className="mb-4">
              <p className="mb-2">From {metroJourney.startStation.name} to {metroJourney.endStation.name}</p>
              <p className="text-sm text-gray-600">Estimated Time: {formatTime(metroJourney.metroTime)}</p>
              <p className="text-sm text-gray-600">Metro Fare: ₹{metroJourney.metroFare.toFixed(2)}</p>
            </div>
            <Map 
              pickup={{
                latitude: metroJourney.startStation.latitude,
                longitude: metroJourney.startStation.longitude,
                address: metroJourney.startStation.name
              }}
              destination={{
                latitude: metroJourney.endStation.latitude,
                longitude: metroJourney.endStation.longitude,
                address: metroJourney.endStation.name
              }}
              className="mb-4"
              showHeatmap={false}
              drivers={[]}
            />
            <div className="space-y-2">
              <button
                onClick={() => window.open(`https://wa.me/${WHATSAPP_METRO_NUMBER}?text=Hi, I would like to book a metro ticket from ${metroJourney.startStation?.name} to ${metroJourney.endStation?.name}. Journey time: ${formatTime(metroJourney.metroTime)}. Fare: ₹${metroJourney.metroFare.toFixed(2)}`, '_blank')}
                className="bg-green-500 text-white px-4 py-2 rounded mb-2 w-full flex items-center justify-center gap-2"
              >
                <span>Book Metro Ticket via WhatsApp</span>
                <span className="text-sm">(₹{metroJourney.metroFare.toFixed(2)})</span>
              </button>
              <button
                onClick={handleMetroRideComplete}
                className="bg-blue-500 text-white px-4 py-2 rounded w-full"
              >
                Metro Ride Complete
              </button>
            </div>
          </div>
        </div>
      )}

      {step === "metroyatri_last_ride" && metroJourney.endStation && destination && (
        <div>
          <h2 className="text-xl font-bold mb-4">Final Leg: To Destination</h2>
          <div className="mb-4 p-4 bg-blue-50 rounded-lg">
            <p className="mb-2">Ride from {metroJourney.endStation.name} Metro Station to {destination.address}</p>
            <p className="text-sm text-gray-600">Estimated Time: {formatTime(metroJourney.lastLegTime)}</p>
            <p className="text-sm text-gray-600">Auto Fare: ₹{metroJourney.lastLegFare.toFixed(2)}</p>
          </div>
          <Map 
            pickup={{
              latitude: metroJourney.endStation.latitude,
              longitude: metroJourney.endStation.longitude,
              address: metroJourney.endStation.name
            }}
            destination={destination}
            driverLocation={driverLocation}
            className="mb-4"
            showHeatmap={false}
            drivers={availableDrivers}
          />
          <DriverMatch 
            drivers={nearestDrivers}
            onDriverConfirmed={handleLastLegDriverConfirmed}
            onCancel={() => setStep("metroyatri_metro")}
          />
        </div>
      )}

      {step === "metroyatri_summary" && metroJourney.startStation && metroJourney.endStation && metroJourney.firstRideDriver && metroJourney.lastRideDriver && (
        <div className="p-4 space-y-4">
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="text-xl font-bold mb-4">Journey Summary</h2>
            
            <div className="space-y-4">
              <div className="border-b pb-4">
                <h3 className="font-semibold mb-2">First Leg</h3>
                <p>From: {pickup.address}</p>
                <p>To: {metroJourney.startStation.name} Metro Station</p>
                <p>Driver: {metroJourney.firstRideDriver.name}</p>
                <p className="text-sm text-gray-600">Time: {formatTime(metroJourney.firstLegTime)}</p>
                <p className="text-sm text-gray-600">Fare: ₹{metroJourney.firstLegFare.toFixed(2)}</p>
              </div>

              <div className="border-b pb-4">
                <h3 className="font-semibold mb-2">Metro Journey</h3>
                <p>From: {metroJourney.startStation.name}</p>
                <p>To: {metroJourney.endStation.name}</p>
                <p className="text-sm text-gray-600">Time: {formatTime(metroJourney.metroTime)}</p>
                <p className="text-sm text-gray-600">Fare: ₹{metroJourney.metroFare.toFixed(2)}</p>
              </div>

              <div className="border-b pb-4">
                <h3 className="font-semibold mb-2">Final Leg</h3>
                <p>From: {metroJourney.endStation.name} Metro Station</p>
                <p>To: {destination?.address}</p>
                <p>Driver: {metroJourney.lastRideDriver.name}</p>
                <p className="text-sm text-gray-600">Time: {formatTime(metroJourney.lastLegTime)}</p>
                <p className="text-sm text-gray-600">Fare: ₹{metroJourney.lastLegFare.toFixed(2)}</p>
              </div>

              <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                <h3 className="font-semibold mb-2">Total Journey</h3>
                <p>Total Time: {formatTime(metroJourney.firstLegTime + metroJourney.metroTime + metroJourney.lastLegTime)}</p>
                <p>Total Fare: ₹{(metroJourney.firstLegFare + metroJourney.metroFare + metroJourney.lastLegFare).toFixed(2)}</p>
              </div>
            </div>

            <Map 
              pickup={pickup}
              destination={destination}
              waypoints={[
                {
                  latitude: metroJourney.startStation.latitude,
                  longitude: metroJourney.startStation.longitude,
                  address: metroJourney.startStation.name
                },
                {
                  latitude: metroJourney.endStation.latitude,
                  longitude: metroJourney.endStation.longitude,
                  address: metroJourney.endStation.name
                }
              ]}
              className="my-4"
              showHeatmap={false}
              drivers={[]}
            />

            <button
              onClick={handleJourneyComplete}
              className="mt-6 bg-green-500 text-white px-4 py-2 rounded w-full"
            >
              Complete Journey
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default RideBooking;
