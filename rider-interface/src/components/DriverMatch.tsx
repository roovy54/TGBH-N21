
import React, { useState, useEffect } from "react";
import { Loader2, Star, Check, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { formatTime } from "@/utils/rideUtils";

interface Driver {
  id: string;
  name: string;
  photo: string;
  rating: number;
  vehicleNumber: string;
  distance: number;
  eta: number;
}

interface DriverMatchProps {
  drivers?: Driver[];
  onDriverConfirmed: (driver: Driver) => void;
  onCancel: () => void;
  className?: string;
}

const DriverMatch: React.FC<DriverMatchProps> = ({ 
  drivers = [], 
  onDriverConfirmed, 
  onCancel,
  className 
}) => {
  const [loading, setLoading] = useState(true);
  const [selectedDriver, setSelectedDriver] = useState<Driver | null>(null);
  const [timeRemaining, setTimeRemaining] = useState(15); // 15 seconds countdown

  // Simulate driver matching process
  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false);
      // Auto-select the nearest driver
      if (drivers && drivers.length > 0) {
        setSelectedDriver(drivers[0]);
      }
    }, 3000);
    
    return () => clearTimeout(timer);
  }, [drivers]);

  // Countdown timer for driver acceptance
  useEffect(() => {
    if (!loading && selectedDriver && timeRemaining > 0) {
      const interval = setInterval(() => {
        setTimeRemaining(prev => prev - 1);
      }, 1000);
      
      return () => clearInterval(interval);
    }
    
    if (!loading && selectedDriver && timeRemaining === 0) {
      onDriverConfirmed(selectedDriver);
    }
  }, [loading, selectedDriver, timeRemaining, onDriverConfirmed]);

  const handleDriverAccept = () => {
    if (selectedDriver) {
      onDriverConfirmed(selectedDriver);
    }
  };

  return (
    <div className={cn("w-full animate-fade-in", className)}>
      <div className="bg-white rounded-xl shadow-md border overflow-hidden">
        <div className="p-4 border-b">
          <h2 className="font-semibold text-lg">
            {loading ? "Finding your driver..." : "Driver found!"}
          </h2>
        </div>
        
        <div className="p-4">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-8">
              <Loader2 className="h-12 w-12 text-namma-blue animate-spin mb-4" />
              <p className="text-muted-foreground">Matching you with the nearest driver</p>
            </div>
          ) : selectedDriver ? (
            <div className="space-y-4">
              <div className="flex justify-between items-start">
                <div className="flex items-center">
                  <div className="w-16 h-16 rounded-full overflow-hidden mr-3 border-2 border-namma-blue">
                    <img 
                      src={selectedDriver.photo} 
                      alt={selectedDriver.name}
                      className="w-full h-full object-cover" 
                    />
                  </div>
                  <div>
                    <h3 className="font-semibold">{selectedDriver.name}</h3>
                    <div className="flex items-center text-sm">
                      <Star className="h-4 w-4 text-yellow-500 fill-yellow-500 mr-1" />
                      <span>{selectedDriver.rating.toFixed(1)}</span>
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">{selectedDriver.vehicleNumber}</p>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="text-sm font-medium text-namma-blue">
                    {selectedDriver.distance.toFixed(1)} km away
                  </div>
                  <div className="text-sm text-muted-foreground">
                    ETA: {formatTime(selectedDriver.eta)}
                  </div>
                </div>
              </div>
              
              <div className="bg-muted/30 p-3 rounded-lg">
                <div className="flex justify-between items-center">
                  <span className="text-sm">Driver is accepting your request</span>
                  <span className="font-medium text-namma-blue">{timeRemaining}s</span>
                </div>
                <div className="w-full bg-muted rounded-full h-2 mt-2">
                  <div
                    className="bg-namma-blue h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${(1 - timeRemaining / 15) * 100}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="flex space-x-3">
                <Button
                  variant="outline"
                  className="flex-1 border-red-500 text-red-500 hover:bg-red-50 hover:text-red-600"
                  onClick={onCancel}
                >
                  <X className="h-4 w-4 mr-2" />
                  Cancel
                </Button>
                <Button
                  className="flex-1 bg-namma-blue hover:bg-namma-blue/90 text-white"
                  onClick={handleDriverAccept}
                >
                  <Check className="h-4 w-4 mr-2" />
                  Accept
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8">
              <p className="text-muted-foreground mb-4">No drivers available at the moment</p>
              <Button onClick={onCancel}>Try Again</Button>
            </div>
          )}
        </div>
      </div>
      
      {!loading && drivers && drivers.length > 1 && (
        <div className="mt-4">
          <h3 className="text-sm font-medium mb-2">Other nearby drivers</h3>
          <div className="space-y-2">
            {drivers.slice(1).map((driver) => (
              <div 
                key={driver.id}
                className="bg-white rounded-lg p-3 border flex items-center justify-between"
              >
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-full overflow-hidden mr-3">
                    <img 
                      src={driver.photo} 
                      alt={driver.name}
                      className="w-full h-full object-cover" 
                    />
                  </div>
                  <div>
                    <div className="font-medium text-sm">{driver.name}</div>
                    <div className="flex items-center text-xs">
                      <Star className="h-3 w-3 text-yellow-500 fill-yellow-500 mr-1" />
                      <span>{driver.rating.toFixed(1)}</span>
                    </div>
                  </div>
                </div>
                <div className="text-right text-xs">
                  <div>{driver.distance.toFixed(1)} km</div>
                  <div className="text-muted-foreground">{formatTime(driver.eta)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DriverMatch;
