// Dummy GPS coordinates (latitude, longitude) for various entities

// User's current location (Bangalore, MG Road area)
export const userLocation = {
  latitude: 12.9716,
  longitude: 77.5946,
  address: "MG Road, Bangalore"
};

// Available destinations
export const popularDestinations = [
  { 
    id: "dest1",
    name: "Indiranagar", 
    latitude: 12.9784, 
    longitude: 77.6408,
    address: "Indiranagar, Bangalore" 
  },
  { 
    id: "dest2", 
    name: "Koramangala", 
    latitude: 12.9352, 
    longitude: 77.6245,
    address: "Koramangala, Bangalore" 
  },
  { 
    id: "dest3", 
    name: "Electronic City", 
    latitude: 12.8399, 
    longitude: 77.6770,
    address: "Electronic City, Bangalore" 
  },
  { 
    id: "dest4", 
    name: "Whitefield", 
    latitude: 12.9698, 
    longitude: 77.7499,
    address: "Whitefield, Bangalore" 
  },
  { 
    id: "dest5", 
    name: "Airport", 
    latitude: 13.1989, 
    longitude: 77.7068,
    address: "Kempegowda International Airport, Bangalore" 
  }
];

// Available vehicle types
export const vehicleTypes = [
  { id: "bike", name: "Bike", image: "/bike-icon.png" },
  { id: "auto", name: "Auto", image: "/auto-icon.png" },
  { id: "cab", name: "Cab", image: "/cab-icon.png" },
  { id: "carpool", name: "Carpool", image: "/carpool-icon.png" },
  { id: "metroyatri", name: "Metro Yatri", image: "/metro-icon.png" } // new Metro Yatri option
];

// Available drivers with their current location
export const availableDrivers = [
  // Bike drivers
  {
    id: "b1",
    name: "Rahul Kumar",
    vehicleType: "bike",
    vehicleNumber: "KA-01-AB-1234",
    rating: 4.7,
    latitude: 12.9726,  // ~0.5 km from user
    longitude: 77.5956,
    photo: "/placeholder.svg",
    isAvailable: true,
    onlineTime: "3h 24m"
  },
  {
    id: "b2",
    name: "Vijay Singh",
    vehicleType: "bike",
    vehicleNumber: "KA-01-CD-5678",
    rating: 4.8,
    latitude: 12.9736,  // ~0.8 km from user
    longitude: 77.5976,
    photo: "/placeholder.svg",
    isAvailable: true,
    onlineTime: "1h 45m"
  },
  {
    id: "b3",
    name: "Kiran Rao",
    vehicleType: "bike",
    vehicleNumber: "KA-01-EF-9012",
    rating: 4.5,
    latitude: 12.9756,  // ~1.2 km from user
    longitude: 77.5986,
    photo: "/placeholder.svg",
    isAvailable: true,
    onlineTime: "4h 10m"
  },
  
  // Auto drivers
  {
    id: "a1",
    name: "Nagesh Kumar",
    vehicleType: "auto",
    vehicleNumber: "KA-01-GH-3456",
    rating: 4.6,
    latitude: 12.9726,  // ~0.5 km from user
    longitude: 77.5976,
    photo: "/placeholder.svg",
    isAvailable: true,
    onlineTime: "2h 30m"
  },
  {
    id: "a2",
    name: "Manoj Patel",
    vehicleType: "auto",
    vehicleNumber: "KA-01-IJ-7890",
    rating: 4.9,
    latitude: 12.9746,  // ~0.9 km from user
    longitude: 77.5926,
    photo: "/placeholder.svg",
    isAvailable: true,
    onlineTime: "5h 15m"
  },
  {
    id: "a3",
    name: "Suresh Babu",
    vehicleType: "auto",
    vehicleNumber: "KA-01-KL-1234",
    rating: 4.7,
    latitude: 12.9686,  // ~1.1 km from user
    longitude: 77.5936,
    photo: "/placeholder.svg",
    isAvailable: true,
    onlineTime: "3h 45m"
  },
  
  // Cab drivers
  {
    id: "c1",
    name: "Ravi Kiran",
    vehicleType: "cab",
    vehicleNumber: "KA-01-MN-5678",
    rating: 4.8,
    latitude: 12.9756,  // ~1.2 km from user
    longitude: 77.5926,
    photo: "/placeholder.svg",
    isAvailable: true,
    onlineTime: "1h 20m"
  },
  {
    id: "c2",
    name: "Deepak Sharma",
    vehicleType: "cab",
    vehicleNumber: "KA-01-OP-9012",
    rating: 4.6,
    latitude: 12.9666,  // ~1.4 km from user
    longitude: 77.5966,
    photo: "/placeholder.svg",
    isAvailable: true,
    onlineTime: "2h 10m"
  },
  {
    id: "c3",
    name: "Akash Reddy",
    vehicleType: "cab",
    vehicleNumber: "KA-01-QR-3456",
    rating: 4.9,
    latitude: 12.9766,  // ~1.5 km from user
    longitude: 77.5906,
    photo: "/placeholder.svg",
    isAvailable: true,
    onlineTime: "4h 5m"
  },
  
  // Carpool drivers
  {
    id: "cp1",
    name: "Amit Sharma",
    vehicleType: "carpool",
    vehicleNumber: "KA-01-CP-0001",
    rating: 4.6,
    latitude: 12.9700,
    longitude: 77.5940,
    photo: "/placeholder.svg",
    isAvailable: true,
    onlineTime: "2h 10m"
  }
];
