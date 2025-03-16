
// Utility functions for ride calculations

/**
 * Calculate the distance between two GPS coordinates using the Haversine formula
 * @returns Distance in kilometers
 */
export const calculateDistance = (
  lat1: number, 
  lon1: number, 
  lat2: number, 
  lon2: number
): number => {
  const R = 6371; // Radius of the earth in km
  const dLat = deg2rad(lat2 - lat1);
  const dLon = deg2rad(lon2 - lon1); 
  
  const a = 
    Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) * 
    Math.sin(dLon/2) * Math.sin(dLon/2); 
  
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)); 
  const distance = R * c; // Distance in km
  
  return parseFloat(distance.toFixed(2));
};

const deg2rad = (deg: number): number => {
  return deg * (Math.PI/180);
};

/**
 * Calculate the estimated time of arrival based on distance
 * @param distance Distance in kilometers
 * @param vehicleType Type of vehicle
 * @returns ETA in minutes
 */
export const calculateETA = (distance: number, vehicleType: string): number => {
  // Average speeds (km/h) based on vehicle type
  const speeds = {
    bike: 30,
    auto: 25,
    cab: 35
  };
  
  // Default to auto speed if vehicle type is not found
  const speed = speeds[vehicleType as keyof typeof speeds] || speeds.auto;
  
  // Calculate time in minutes: (distance / speed) * 60
  // Add some buffer time based on traffic conditions
  const baseTime = (distance / speed) * 60;
  const trafficBuffer = 1.2; // 20% buffer for traffic
  
  return Math.round(baseTime * trafficBuffer);
};

/**
 * Calculate the fare for a ride
 * @param distance Distance in kilometers
 * @param duration Estimated duration in minutes
 * @param vehicleType Type of vehicle
 * @returns Fare in rupees
 */
export const calculateFare = (
  distance: number, 
  duration: number, 
  vehicleType: string
): number => {
  // Base prices and rates
  const prices = {
    bike: { base: 20, perKm: 7, perMinute: 1 },
    auto: { base: 30, perKm: 13, perMinute: 1.5 },
    cab: { base: 50, perKm: 18, perMinute: 2 }
  };
  
  // Default to auto pricing if vehicle type is not found
  const pricing = prices[vehicleType as keyof typeof prices] || prices.auto;
  
  // Calculate fare: base price + (distance * price per km) + (duration * price per minute)
  const baseFare = pricing.base;
  const distanceFare = distance * pricing.perKm;
  const durationFare = duration * pricing.perMinute;
  
  // Apply minimum fare and round to the nearest integer
  const totalFare = baseFare + distanceFare + durationFare;
  const minimumFare = pricing.base * 1.5;
  
  return Math.max(Math.round(totalFare), Math.round(minimumFare));
};

/**
 * Find the nearest available drivers based on user's location
 * @param userLat User's latitude
 * @param userLon User's longitude
 * @param drivers Array of available drivers
 * @param vehicleType Type of vehicle to filter (optional)
 * @param limit Maximum number of drivers to return
 * @returns Array of nearest drivers with calculated distance and ETA
 */
export const findNearestDrivers = (
  userLat: number,
  userLon: number,
  drivers: any[],
  vehicleType?: string,
  limit: number = 3
) => {
  // Filter drivers by vehicle type if specified
  const filteredDrivers = vehicleType 
    ? drivers.filter(driver => driver.vehicleType === vehicleType && driver.isAvailable) 
    : drivers.filter(driver => driver.isAvailable);
  
  // Calculate distance for each driver
  const driversWithDistance = filteredDrivers.map(driver => {
    const distance = calculateDistance(userLat, userLon, driver.latitude, driver.longitude);
    const eta = calculateETA(distance, driver.vehicleType);
    
    return {
      ...driver,
      distance,
      eta
    };
  });
  
  // Sort by distance (nearest first) and limit the results
  return driversWithDistance
    .sort((a, b) => a.distance - b.distance)
    .slice(0, limit);
};

/**
 * Format time in minutes to a human-readable string
 */
export const formatTime = (minutes: number): string => {
  if (minutes < 60) {
    return `${minutes} min`;
  }
  
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  
  if (mins === 0) {
    return `${hours} hour${hours > 1 ? 's' : ''}`;
  }
  
  return `${hours} hour${hours > 1 ? 's' : ''} ${mins} min`;
};

/**
 * Generate a ride tracking path based on start and end points
 * @returns Array of coordinates representing the path
 */
export const generateTrackingPath = (
  startLat: number,
  startLon: number,
  endLat: number,
  endLon: number,
  numPoints: number = 10
): { latitude: number, longitude: number }[] => {
  const result = [];
  
  for (let i = 0; i < numPoints; i++) {
    const ratio = i / (numPoints - 1);
    const lat = startLat + (endLat - startLat) * ratio;
    const lon = startLon + (endLon - startLon) * ratio;
    
    result.push({
      latitude: lat,
      longitude: lon
    });
  }
  
  return result;
};

// This function implements a simple batched matching algorithm.
// It expands the search radius until at least maxDrivers are found (or until a max radius is reached),
// then sorts drivers based on ETA (primary) and rating (secondary).
export function batchedMatchDrivers(
  pickupLat: number,
  pickupLng: number,
  drivers: any[],
  vehicleType: string,
  maxDrivers: number = 3,
  initialRadius: number = 2 // in kilometers
) {
  let radius = initialRadius;
  let matchedDrivers: any[] = [];
  const maxRadius = 10; // set an upper limit for search radius

  // Expand search radius until we have enough drivers or we hit the max radius
  while (radius <= maxRadius) {
    matchedDrivers = drivers.filter((driver) => {
      // Ensure driver is of the requested vehicle type
      if (driver.vehicleType !== vehicleType) return false;
      const dist = calculateDistance(pickupLat, pickupLng, driver.latitude, driver.longitude);
      return dist <= radius;
    });
    if (matchedDrivers.length >= maxDrivers) {
      break;
    }
    radius += 2; // increase radius by 2 km
  }

  // For each matched driver, calculate current distance and ETA
  matchedDrivers = matchedDrivers.map((driver) => {
    const dist = calculateDistance(pickupLat, pickupLng, driver.latitude, driver.longitude);
    return {
      ...driver,
      distance: dist,
      eta: calculateETA(dist, vehicleType)
    };
  });

  // Sort drivers: primary sort by ETA ascending; if equal, sort by rating descending.
  matchedDrivers.sort((a, b) => a.eta - b.eta || b.rating - a.rating);

  // Return up to maxDrivers drivers
  return matchedDrivers.slice(0, maxDrivers);
}
