
import React, { useState } from "react";
import { Bike, Car, ChevronRight, Clock, MoreHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { vehicleTypes } from "@/data/dummyData";

interface RideOption {
  id: string;
  name: string;
  image: string;
  estimated_time: string;
  price: number;
  promocode?: string;
  discount?: number;
}

interface RideOptionsProps {
  options: RideOption[];
  onSelect: (option: RideOption) => void;
  className?: string;
  distance?: number; // Distance in kilometers
}

const RideOptions: React.FC<RideOptionsProps> = ({ 
  options, 
  onSelect, 
  className,
  distance = 5 // Default distance if not provided
}) => {
  const [selectedOption, setSelectedOption] = useState<string | null>(null);

  const handleSelect = (option: RideOption) => {
    setSelectedOption(option.id);
    onSelect(option);
  };

  const getVehicleIcon = (vehicleId: string) => {
    switch (vehicleId) {
      case "bike":
        return <Bike className="h-6 w-6" />;
      case "auto":
        return (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="h-6 w-6">
            <path d="M7 17m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" stroke="currentColor" strokeWidth="2"/>
            <path d="M17 17m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" stroke="currentColor" strokeWidth="2"/>
            <path d="M5 17h-2v-6l2 -5h9l4 5h1a2 2 0 0 1 2 2v4h-2m-4 0h-6m-6 -6h15m-6 0v-5" stroke="currentColor" strokeWidth="2"/>
          </svg>
        );
      case "cab":
        return <Car className="h-6 w-6" />;
      default:
        return <MoreHorizontal className="h-6 w-6" />;
    }
  };

  return (
    <div className={cn("w-full animate-slide-up", className)}>
      <div className="bg-white rounded-xl shadow-md border overflow-hidden">
        <div className="p-4 border-b">
          <h2 className="font-semibold text-lg">Select Ride</h2>
          {distance && (
            <p className="text-sm text-muted-foreground mt-1">
              Distance: {distance} km
            </p>
          )}
        </div>
        
        <div className="divide-y">
          {options.map((option) => (
            <div
              key={option.id}
              className={cn(
                "p-4 transition-colors hover:bg-muted/50",
                selectedOption === option.id && "bg-muted/50"
              )}
            >
              <button 
                className="w-full flex items-start justify-between"
                onClick={() => handleSelect(option)}
              >
                <div className="flex items-center">
                  <div className="w-16 h-16 bg-muted rounded-lg overflow-hidden flex items-center justify-center mr-3">
                    {getVehicleIcon(option.id)}
                  </div>
                  
                  <div className="text-left">
                    <h3 className="font-semibold text-base">{option.name}</h3>
                    <div className="flex items-center text-sm text-muted-foreground mt-1">
                      <Clock className="h-3.5 w-3.5 mr-1" />
                      <span>{option.estimated_time}</span>
                    </div>
                    {option.promocode && (
                      <div className="mt-1">
                        <span className="text-xs px-2 py-0.5 rounded-full bg-accent text-primary-foreground font-medium">
                          {option.promocode}: {option.discount}% off
                        </span>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="flex flex-col items-end">
                  <span className="font-semibold">₹{option.price}</span>
                  {option.discount && (
                    <span className="text-xs text-muted-foreground line-through">
                      ₹{Math.round(option.price / (1 - option.discount / 100))}
                    </span>
                  )}
                  <ChevronRight className="h-5 w-5 text-muted-foreground mt-2" />
                </div>
              </button>
            </div>
          ))}
        </div>
        
        <div className="p-4 bg-muted/30">
          <Button 
            className="w-full py-6 bg-namma-blue hover:bg-namma-blue/90 text-white font-medium"
            disabled={!selectedOption}
            onClick={() => {
              const selected = options.find(o => o.id === selectedOption);
              if (selected) onSelect(selected);
            }}
          >
            Confirm {selectedOption ? options.find(o => o.id === selectedOption)?.name : "Ride"}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default RideOptions;
