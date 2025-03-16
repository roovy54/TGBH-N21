
import React from "react";
import { useParams, useNavigate } from "react-router-dom";
import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ChevronRight, Star, MapPin, Navigation, Clock, Calendar, Phone, MessageSquare, User } from "lucide-react";
import { Separator } from "@/components/ui/separator";

const RideDetails: React.FC = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  
  // This would normally come from an API call using the ID
  const ride = {
    id: id || "ride-123",
    status: "completed",
    date: "May 15, 2023",
    time: "14:30",
    pickup: "MG Road Metro Station",
    destination: "Indiranagar, Bangalore",
    distance: "5.2 km",
    duration: "22 min",
    fare: 140,
    driver: {
      name: "Suresh Kumar",
      photo: "https://randomuser.me/api/portraits/men/32.jpg",
      vehicle: "KA 05 MJ 6789",
      rating: 4.8
    }
  };

  return (
    <Layout>
      <div className="max-w-md mx-auto">
        <Button
          variant="ghost"
          size="icon"
          className="mb-4"
          onClick={() => navigate(-1)}
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>
        
        <div className="bg-white rounded-xl shadow-md border overflow-hidden animate-fade-in">
          <div className="p-4 border-b flex justify-between items-center">
            <h2 className="font-semibold text-lg">Ride Details</h2>
            <div className={
              ride.status === "completed" 
                ? "px-2.5 py-1 text-xs rounded-full bg-namma-green/10 text-namma-green font-medium" 
                : "px-2.5 py-1 text-xs rounded-full bg-destructive/10 text-destructive font-medium"
            }>
              {ride.status === "completed" ? "Completed" : "Cancelled"}
            </div>
          </div>
          
          <div className="p-4">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <Calendar className="h-4 w-4 mr-2 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">{ride.date}</span>
              </div>
              <div className="flex items-center">
                <Clock className="h-4 w-4 mr-2 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">{ride.time}</span>
              </div>
            </div>
            
            <div className="space-y-6 mb-6">
              {/* Pickup location */}
              <div className="flex items-start">
                <div className="w-8 flex-shrink-0 flex justify-center">
                  <div className="w-8 h-8 bg-namma-blue/10 rounded-full flex items-center justify-center">
                    <MapPin className="h-4 w-4 text-namma-blue" />
                  </div>
                </div>
                <div className="ml-3">
                  <span className="text-sm text-muted-foreground">Pickup</span>
                  <p className="font-medium">{ride.pickup}</p>
                </div>
              </div>
              
              {/* Line connecting pickup and destination */}
              <div className="flex items-center">
                <div className="w-8 flex-shrink-0 flex justify-center">
                  <div className="w-0.5 h-12 bg-namma-blue/30"></div>
                </div>
              </div>
              
              {/* Destination */}
              <div className="flex items-start">
                <div className="w-8 flex-shrink-0 flex justify-center">
                  <div className="w-8 h-8 bg-namma-blue/10 rounded-full flex items-center justify-center">
                    <Navigation className="h-4 w-4 text-namma-blue" />
                  </div>
                </div>
                <div className="ml-3">
                  <span className="text-sm text-muted-foreground">Destination</span>
                  <p className="font-medium">{ride.destination}</p>
                </div>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-3 mb-6">
              <div className="bg-muted/30 p-3 rounded-lg">
                <span className="text-xs text-muted-foreground">Distance</span>
                <p className="font-semibold">{ride.distance}</p>
              </div>
              <div className="bg-muted/30 p-3 rounded-lg">
                <span className="text-xs text-muted-foreground">Duration</span>
                <p className="font-semibold">{ride.duration}</p>
              </div>
              <div className="bg-muted/30 p-3 rounded-lg">
                <span className="text-xs text-muted-foreground">Fare</span>
                <p className="font-semibold">₹{ride.fare}</p>
              </div>
              <div className="bg-muted/30 p-3 rounded-lg">
                <span className="text-xs text-muted-foreground">Payment</span>
                <p className="font-semibold">Cash</p>
              </div>
            </div>
            
            <Separator className="my-6" />
            
            <div>
              <h3 className="font-semibold mb-3">Driver Information</h3>
              
              <div className="flex items-center">
                <div className="w-14 h-14 rounded-full overflow-hidden border-2 border-namma-blue mr-3">
                  <img 
                    src={ride.driver.photo} 
                    alt={ride.driver.name}
                    className="w-full h-full object-cover" 
                  />
                </div>
                
                <div className="flex-1">
                  <div className="flex items-center">
                    <h4 className="font-medium">{ride.driver.name}</h4>
                    <div className="ml-2 flex items-center text-sm bg-namma-green/10 text-namma-green px-2 py-0.5 rounded-full">
                      <Star className="h-3 w-3 mr-1 fill-namma-green text-namma-green" />
                      {ride.driver.rating}
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Auto • {ride.driver.vehicle}
                  </p>
                </div>
              </div>
              
              <div className="flex mt-4 space-x-3">
                <Button
                  variant="outline"
                  className="flex-1 border-namma-blue/30 hover:bg-namma-blue/5 text-namma-blue"
                  onClick={() => alert("Calling driver...")}
                >
                  <Phone className="h-4 w-4 mr-2" />
                  Call
                </Button>
                <Button
                  variant="outline"
                  className="flex-1 border-namma-blue/30 hover:bg-namma-blue/5 text-namma-blue"
                  onClick={() => alert("Messaging driver...")}
                >
                  <MessageSquare className="h-4 w-4 mr-2" />
                  Message
                </Button>
              </div>
            </div>
            
            <Button
              className="w-full mt-6 bg-namma-blue hover:bg-namma-blue/90 text-white"
              onClick={() => navigate("/")}
            >
              Book Another Ride
            </Button>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default RideDetails;
