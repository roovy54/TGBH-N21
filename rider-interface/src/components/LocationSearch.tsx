
import React, { useState } from "react";
import { MapPin, Navigation, X, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface LocationSearchProps {
  onLocationsSelected: (pickup: string, destination: string) => void;
  className?: string;
}

const LocationSearch: React.FC<LocationSearchProps> = ({ 
  onLocationsSelected,
  className 
}) => {
  const [pickup, setPickup] = useState("");
  const [destination, setDestination] = useState("");
  const [isFocused, setIsFocused] = useState<"pickup" | "destination" | null>(null);
  const [recentLocations] = useState([
    "Home - 123 Main St",
    "Work - 456 Office Park",
    "Gym - Fitness Center",
    "Mall - Shopping Plaza"
  ]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (pickup && destination) {
      onLocationsSelected(pickup, destination);
    }
  };

  const handleRecentLocationClick = (location: string) => {
    if (isFocused === "pickup") {
      setPickup(location);
    } else if (isFocused === "destination") {
      setDestination(location);
    }
    setIsFocused(null);
  };

  const clearInput = (type: "pickup" | "destination") => {
    if (type === "pickup") {
      setPickup("");
    } else {
      setDestination("");
    }
  };

  return (
    <div className={cn("w-full", className)}>
      <form 
        onSubmit={handleSubmit}
        className="relative bg-white rounded-xl shadow-md border p-4 animate-slide-up transition-all"
      >
        <div className="space-y-3">
          {/* Pickup Location Input */}
          <div className="relative flex items-center">
            <div className="absolute left-3 flex h-full items-center">
              <MapPin className="h-5 w-5 text-namma-blue" />
            </div>
            <Input
              type="text"
              placeholder="Pickup location"
              className="pl-10 pr-10 py-6 bg-muted/50 border-0 rounded-lg focus-visible:ring-1 focus-visible:ring-namma-blue"
              value={pickup}
              onChange={(e) => setPickup(e.target.value)}
              onFocus={() => setIsFocused("pickup")}
            />
            {pickup && (
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="absolute right-2"
                onClick={() => clearInput("pickup")}
              >
                <X className="h-4 w-4 text-muted-foreground" />
              </Button>
            )}
          </div>

          {/* Destination Input */}
          <div className="relative flex items-center">
            <div className="absolute left-3 flex h-full items-center">
              <Navigation className="h-5 w-5 text-namma-blue" />
            </div>
            <Input
              type="text"
              placeholder="Where to?"
              className="pl-10 pr-10 py-6 bg-muted/50 border-0 rounded-lg focus-visible:ring-1 focus-visible:ring-namma-blue"
              value={destination}
              onChange={(e) => setDestination(e.target.value)}
              onFocus={() => setIsFocused("destination")}
            />
            {destination && (
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="absolute right-2"
                onClick={() => clearInput("destination")}
              >
                <X className="h-4 w-4 text-muted-foreground" />
              </Button>
            )}
          </div>
          
          <Button 
            type="submit" 
            className="w-full py-6 bg-namma-blue hover:bg-namma-blue/90 text-white font-medium"
            disabled={!pickup || !destination}
          >
            Search Rides
          </Button>
        </div>

        {/* Recent Locations */}
        {isFocused && (
          <div className="absolute top-full left-0 right-0 mt-2 bg-white rounded-lg shadow-lg border p-2 z-10 animate-slide-down">
            <div className="flex items-center px-3 py-2 text-sm text-muted-foreground">
              <Search className="h-4 w-4 mr-2" />
              <span>Recent locations</span>
            </div>
            <ul className="space-y-1">
              {recentLocations.map((location, index) => (
                <li key={index}>
                  <button
                    type="button"
                    className="w-full text-left px-3 py-2.5 rounded-md text-sm hover:bg-muted flex items-center space-x-2"
                    onClick={() => handleRecentLocationClick(location)}
                  >
                    <MapPin className="h-4 w-4 text-muted-foreground" />
                    <span>{location}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </form>
    </div>
  );
};

export default LocationSearch;
