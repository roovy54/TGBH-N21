
import React from "react";
import RideBooking from "@/components/RideBooking";
import Layout from "@/components/Layout";

const Index: React.FC = () => {
  return (
    <Layout>
      <div className="max-w-7xl mx-auto w-full">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left column - Ride Booking */}
          <div className="flex flex-col items-center">
            <div className="mb-8 text-center">
              <div className="bg-namma-blue text-white inline-block px-2 py-1 rounded-md text-xs font-medium mb-2">NAMMA YATRI</div>
              <h1 className="text-3xl font-bold text-namma-dark mb-2">Book Your Auto Ride</h1>
              <p className="text-muted-foreground">Fast, reliable, and affordable transportation</p>
            </div>
            
            <RideBooking />
          </div>
          
          {/* Right column - App Mockup Image */}
          <div className="hidden md:flex items-center justify-center relative">
            <div className="bg-gradient-to-tr from-namma-blue/10 to-namma-blue/5 rounded-2xl w-full aspect-square max-w-md flex items-center justify-center p-8 shadow-sm">
              <div className="relative w-64 h-[500px] bg-white rounded-[36px] shadow-xl overflow-hidden border-8 border-gray-800">
                <div className="absolute top-0 left-0 right-0 h-6 bg-gray-800 flex justify-center items-end pb-1">
                  <div className="w-20 h-1.5 bg-gray-600 rounded-full"></div>
                </div>
                
                <div className="pt-6 h-full bg-gradient-to-b from-namma-blue to-namma-blue/90 flex flex-col items-center text-white p-4">
                  <div className="text-lg font-bold mt-2 animate-pulse-soft">Namma Yatri</div>
                  
                  <div className="mt-10 w-full bg-white/10 backdrop-blur-sm rounded-lg p-3">
                    <div className="mb-2 text-sm font-medium">Your location</div>
                    <div className="flex items-center bg-white/20 rounded-lg p-2 mb-3">
                      <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center mr-2">
                        <div className="w-3 h-3 bg-white rounded-full"></div>
                      </div>
                      <div className="text-xs">Current Location</div>
                    </div>
                    
                    <div className="mb-2 text-sm font-medium">Destination</div>
                    <div className="flex items-center bg-white/20 rounded-lg p-2">
                      <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center mr-2">
                        <div className="w-3 h-3 bg-white rounded-full"></div>
                      </div>
                      <div className="text-xs">Search destination</div>
                    </div>
                  </div>
                  
                  <div className="mt-4 w-full bg-white/10 backdrop-blur-sm rounded-lg p-3">
                    <div className="text-sm font-medium mb-2">Suggested</div>
                    
                    <div className="space-y-2">
                      {["Home", "Work", "Friends House"].map((place, i) => (
                        <div key={i} className="flex items-center bg-white/20 rounded-lg p-2">
                          <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center mr-2">
                            <div className="w-3 h-3 bg-white rounded-full"></div>
                          </div>
                          <div className="text-xs">{place}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div className="mt-auto mb-4 w-full bg-white/90 text-namma-blue font-medium py-3 rounded-lg text-center text-sm">
                    Book Now
                  </div>
                </div>
              </div>
            </div>
            
            <div className="absolute -z-10 w-full h-full max-w-md max-h-md blur-3xl rounded-full bg-namma-blue/20 animate-pulse-soft"></div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Index;
