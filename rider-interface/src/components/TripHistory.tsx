
import React from "react";
import { ChevronRight, Clock, Calendar, MapPin, Navigation } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface Trip {
  id: string;
  date: string;
  time: string;
  pickup: string;
  destination: string;
  fare: number;
  driverName: string;
  status: "completed" | "cancelled";
}

interface TripHistoryProps {
  trips: Trip[];
  onTripSelected: (trip: Trip) => void;
  className?: string;
}

const TripHistory: React.FC<TripHistoryProps> = ({
  trips,
  onTripSelected,
  className
}) => {
  // Group trips by date
  const groupedTrips: Record<string, Trip[]> = {};
  
  trips.forEach(trip => {
    if (!groupedTrips[trip.date]) {
      groupedTrips[trip.date] = [];
    }
    groupedTrips[trip.date].push(trip);
  });

  return (
    <div className={cn("w-full", className)}>
      {Object.entries(groupedTrips).map(([date, dateTrips]) => (
        <div key={date} className="mb-6">
          <div className="flex items-center mb-3">
            <Calendar className="h-4 w-4 mr-2 text-muted-foreground" />
            <h3 className="font-semibold">{date}</h3>
          </div>
          
          <div className="space-y-3">
            {dateTrips.map(trip => (
              <div 
                key={trip.id}
                className="bg-white rounded-xl border shadow-sm overflow-hidden animate-slide-up"
                style={{ animationDelay: `${dateTrips.indexOf(trip) * 50}ms` }}
              >
                <Button 
                  variant="ghost" 
                  className="w-full p-4 h-auto flex items-start justify-between hover:bg-muted/50 text-left"
                  onClick={() => onTripSelected(trip)}
                >
                  <div className="space-y-2">
                    <div className="flex items-center">
                      <Clock className="h-3.5 w-3.5 mr-1.5 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">{trip.time}</span>
                      <div className={cn(
                        "ml-2 px-2 py-0.5 text-xs rounded-full font-medium",
                        trip.status === "completed" 
                          ? "bg-namma-green/10 text-namma-green" 
                          : "bg-destructive/10 text-destructive"
                      )}>
                        {trip.status === "completed" ? "Completed" : "Cancelled"}
                      </div>
                    </div>
                    
                    <div className="space-y-1">
                      <div className="flex items-start">
                        <MapPin className="h-4 w-4 mr-1.5 text-namma-blue mt-0.5" />
                        <span className="text-sm font-medium truncate max-w-[180px]">{trip.pickup}</span>
                      </div>
                      <div className="flex items-start">
                        <Navigation className="h-4 w-4 mr-1.5 text-namma-blue mt-0.5" />
                        <span className="text-sm font-medium truncate max-w-[180px]">{trip.destination}</span>
                      </div>
                    </div>
                    
                    <div className="text-xs text-muted-foreground">
                      Driver: {trip.driverName}
                    </div>
                  </div>
                  
                  <div className="flex flex-col items-end">
                    <span className="font-semibold">₹{trip.fare}</span>
                    <ChevronRight className="h-5 w-5 text-muted-foreground mt-3" />
                  </div>
                </Button>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default TripHistory;
