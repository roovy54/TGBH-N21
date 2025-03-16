import React, { useState } from "react";
import { Button } from "@/components/ui/button";

interface FareBreakdownProps {
  baseFare: number;  // This comes from calculateFare in ride selection!
  rideOption: {
    id: string;
    name: string;
    estimated_time: string;
    price: number;
    promocode?: string;
    discount?: number;
  };
  onAccept: (finalFare: number) => void;
  onDeny: () => void;
}

const extraAmounts = [10, 20, 30, 50];

const FareBreakdown: React.FC<FareBreakdownProps> = ({ baseFare, rideOption, onAccept, onDeny }) => {
  const [extra, setExtra] = useState<number>(0);

  // Use fixed percentages so that fees add up exactly to the computed fare.
  const baseFareComponent = parseFloat((baseFare * 0.30).toFixed(2));     // 30%
  const distanceRate     = parseFloat((baseFare * 0.25).toFixed(2));     // 25%
  const timeRate         = parseFloat((baseFare * 0.20).toFixed(2));     // 20%
  const surgePricing     = parseFloat((baseFare * 0.15).toFixed(2));     // 15%
  const bookingFee       = parseFloat((baseFare * 0.05).toFixed(2));     // 5%
  const tolls            = parseFloat((baseFare * 0.03).toFixed(2));     // 3%
  const surcharges       = parseFloat((baseFare * 0.02).toFixed(2));     // 2%

  // Sum of components equals the passed base fare
  const computedFare = baseFareComponent + distanceRate + timeRate + surgePricing + bookingFee + tolls + surcharges;
  const finalFare = computedFare + extra;

  return (
    <div className="bg-white p-6 shadow-lg rounded-lg max-w-md mx-auto">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Fare Breakdown</h2>
      <p className="text-gray-600 mb-4">Ride Option: <span className="font-medium">{rideOption.name}</span></p>
      <ul className="mb-4 space-y-1 text-gray-700">
        <li className="flex justify-between">
          <span>Base Fare</span>
          <span>₹{baseFareComponent.toFixed(2)}</span>
        </li>
        <li className="flex justify-between">
          <span>Distance Rate</span>
          <span>₹{distanceRate.toFixed(2)}</span>
        </li>
        <li className="flex justify-between">
          <span>Time Rate</span>
          <span>₹{timeRate.toFixed(2)}</span>
        </li>
        <li className="flex justify-between">
          <span>Surge Pricing</span>
          <span>₹{surgePricing.toFixed(2)}</span>
        </li>
        <li className="flex justify-between">
          <span>Booking Fee</span>
          <span>₹{bookingFee.toFixed(2)}</span>
        </li>
        <li className="flex justify-between">
          <span>Tolls</span>
          <span>₹{tolls.toFixed(2)}</span>
        </li>
        <li className="flex justify-between">
          <span>Surcharges</span>
          <span>₹{surcharges.toFixed(2)}</span>
        </li>
      </ul>
      <div className="my-4">
        <p className="font-medium mb-2">Add Extra Amount:</p>
        <div className="flex gap-3">
          {extraAmounts.map((amount) => (
            <Button
              key={amount}
              variant={extra === amount ? "default" : "outline"}
              onClick={() => setExtra(amount)}
              className="py-1 px-3 text-sm"
            >
              +₹{amount}
            </Button>
          ))}
        </div>
      </div>
      <p className="text-xl font-semibold text-gray-800 mb-6">Final Fare: ₹{finalFare.toFixed(2)}</p>
      <div className="flex justify-center gap-6">
        <Button
          className="bg-green-500 hover:bg-green-600 text-white py-2 px-6 rounded-lg"
          onClick={() => onAccept(finalFare)}
        >
          Accept
        </Button>
        <Button
          className="bg-red-500 hover:bg-red-600 text-white py-2 px-6 rounded-lg"
          onClick={onDeny}
        >
          Deny
        </Button>
      </div>
    </div>
  );
};

export default FareBreakdown;