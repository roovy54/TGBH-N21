export interface Location {
  latitude: number;
  longitude: number;
  address: string;
}

export interface Driver {
  id: string;
  name: string;
  vehicleType: string;
  vehicleNumber: string;
  rating: number;
  photo: string;
  latitude: number;
  longitude: number;
  isAvailable: boolean;
  gender: string;
  distance?: number;
  eta?: number;
}

export interface RideOption {
  id: string;
  name: string;
  vehicleType: string;
  price: number;
  eta: number;
  estimated_time: string;
  image: string;
  promocode?: string;
  discount?: number;
}

export type BookingStep = 
  | "home" 
  | "location" 
  | "options" 
  | "fareBreakdown" 
  | "matching" 
  | "tracking" 
  | "schedule" 
  | "scheduled"
  | "metro-yatri"; 