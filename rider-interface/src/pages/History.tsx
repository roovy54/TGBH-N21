
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Layout from "@/components/Layout";
import TripHistory from "@/components/TripHistory";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";

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

const History: React.FC = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState("");
  
  // Sample trip data
  const trips: Trip[] = [
    {
      id: "trip-1",
      date: "Today",
      time: "14:30",
      pickup: "MG Road Metro Station",
      destination: "Indiranagar, Bangalore",
      fare: 140,
      driverName: "Suresh Kumar",
      status: "completed"
    },
    {
      id: "trip-2",
      date: "Today",
      time: "09:15",
      pickup: "Home",
      destination: "Whitefield, Bangalore",
      fare: 320,
      driverName: "Ramesh S",
      status: "completed"
    },
    {
      id: "trip-3",
      date: "Yesterday",
      time: "19:45",
      pickup: "Phoenix Mall, Whitefield",
      destination: "Home",
      fare: 280,
      driverName: "Venkat R",
      status: "completed"
    },
    {
      id: "trip-4",
      date: "Yesterday",
      time: "13:20",
      pickup: "Office",
      destination: "Koramangala, Bangalore",
      fare: 190,
      driverName: "Mahesh K",
      status: "cancelled"
    },
    {
      id: "trip-5",
      date: "May 12, 2023",
      time: "11:10",
      pickup: "Koramangala, Bangalore",
      destination: "Electronic City",
      fare: 350,
      driverName: "Ravi Kumar",
      status: "completed"
    }
  ];

  // Filter trips based on search term
  const filteredTrips = trips.filter(trip => {
    const searchString = searchTerm.toLowerCase();
    return (
      trip.pickup.toLowerCase().includes(searchString) ||
      trip.destination.toLowerCase().includes(searchString) ||
      trip.driverName.toLowerCase().includes(searchString)
    );
  });

  const handleTripSelected = (trip: Trip) => {
    navigate(`/ride/${trip.id}`);
  };

  return (
    <Layout>
      <div className="max-w-md mx-auto">
        <div className="mb-6">
          <h1 className="text-2xl font-bold mb-2">Your Rides</h1>
          <p className="text-muted-foreground">View your ride history and details</p>
        </div>
        
        <div className="relative mb-6">
          <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
            <Search className="h-4 w-4 text-muted-foreground" />
          </div>
          <Input
            type="text"
            placeholder="Search your rides"
            className="pl-10 bg-white"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        
        {filteredTrips.length > 0 ? (
          <TripHistory 
            trips={filteredTrips}
            onTripSelected={handleTripSelected}
          />
        ) : (
          <div className="text-center py-12 bg-white rounded-xl border shadow-sm">
            <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
              <Search className="h-8 w-8 text-muted-foreground" />
            </div>
            <h3 className="font-semibold mb-1">No rides found</h3>
            <p className="text-muted-foreground text-sm">
              {searchTerm
                ? `No rides matching "${searchTerm}"`
                : "You don't have any rides yet"}
            </p>
          </div>
        )}
      </div>
    </Layout>
  );
};

export default History;
