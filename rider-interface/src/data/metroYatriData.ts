// Metro stations in Bangalore
export const metroStations = [
  {
    id: "ms1",
    name: "Majestic",
    latitude: 12.9766,
    longitude: 77.5713,
    address: "Majestic Metro Station, Bangalore"
  },
  {
    id: "ms2",
    name: "MG Road",
    latitude: 12.9758,
    longitude: 77.6180,
    address: "MG Road Metro Station, Bangalore"
  },
  {
    id: "ms3",
    name: "Indiranagar",
    latitude: 12.9784,
    longitude: 77.6408,
    address: "Indiranagar Metro Station, Bangalore"
  },
  {
    id: "ms4",
    name: "Mysore Road",
    latitude: 12.9543,
    longitude: 77.5344,
    address: "Mysore Road Metro Station, Bangalore"
  }
];

// Sample Metro Yatri rides
export const metroYatriRides = [
  {
    id: "my1",
    riderName: "Priya Sharma",
    startLocation: {
      latitude: 12.9352,
      longitude: 77.6245,
      address: "Koramangala, Bangalore"
    },
    nearestMetroStation: {
      id: "ms2",
      name: "MG Road",
      latitude: 12.9758,
      longitude: 77.6180,
      address: "MG Road Metro Station, Bangalore"
    },
    destinationMetroStation: {
      id: "ms4",
      name: "Mysore Road",
      latitude: 12.9543,
      longitude: 77.5344,
      address: "Mysore Road Metro Station, Bangalore"
    },
    finalDestination: {
      latitude: 12.9456,
      longitude: 77.5123,
      address: "RR Nagar, Bangalore"
    }
  },
  {
    id: "my2",
    riderName: "Arun Kumar",
    startLocation: {
      latitude: 12.9698,
      longitude: 77.7499,
      address: "Whitefield, Bangalore"
    },
    nearestMetroStation: {
      id: "ms3",
      name: "Indiranagar",
      latitude: 12.9784,
      longitude: 77.6408,
      address: "Indiranagar Metro Station, Bangalore"
    },
    destinationMetroStation: {
      id: "ms1",
      name: "Majestic",
      latitude: 12.9766,
      longitude: 77.5713,
      address: "Majestic Metro Station, Bangalore"
    },
    finalDestination: {
      latitude: 12.9716,
      longitude: 77.5946,
      address: "City Market, Bangalore"
    }
  },
  {
    id: "my3",
    riderName: "Meera Reddy",
    startLocation: {
      latitude: 12.9989,
      longitude: 77.5920,
      address: "Malleshwaram, Bangalore"
    },
    nearestMetroStation: {
      id: "ms1",
      name: "Majestic",
      latitude: 12.9766,
      longitude: 77.5713,
      address: "Majestic Metro Station, Bangalore"
    },
    destinationMetroStation: {
      id: "ms3",
      name: "Indiranagar",
      latitude: 12.9784,
      longitude: 77.6408,
      address: "Indiranagar Metro Station, Bangalore"
    },
    finalDestination: {
      latitude: 12.9784,
      longitude: 77.6408,
      address: "Defence Colony, Indiranagar, Bangalore"
    }
  }
];

export const WHATSAPP_METRO_NUMBER = "+918105556677"; 