import React, { useState } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { MapPin, Phone, AtSign, CreditCard, Shield, User, Bell, Star, Award } from "lucide-react";
import { cn } from "@/lib/utils";

interface UserProfileProps {
  className?: string;
}

const UserProfile: React.FC<UserProfileProps> = ({ className }) => {
  const [user] = useState({
    name: "Priya Sharma",
    avatar: "https://randomuser.me/api/portraits/women/44.jpg",
    email: "priya.sharma@example.com",
    phone: "+91 98765 43210",
    address: "123 MG Road, Bangalore",
    totalRides: 48,
    rating: 4.9,
    nammaPoints: 560
  });

  const [isEditing, setIsEditing] = useState(false);

  const nammaCoinsOptions = [
    {
      name: "Amazon",
      description: "₹250 Amazon gift card",
      points: 500,
      image: "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Amazon_logo.svg/2560px-Amazon_logo.svg.png"
    },
    {
      name: "Swiggy",
      description: "₹150 off on your next order",
      points: 300,
      image: "https://logos-world.net/wp-content/uploads/2020/11/Swiggy-Logo.png"
    }
  ];

  return (
    <div className={cn("w-full max-w-3xl mx-auto", className)}>
      <div className="bg-white rounded-xl shadow-md border overflow-hidden animate-fade-in">
        <div className="p-6 flex flex-col md:flex-row gap-6">
          <div className="flex flex-col items-center justify-center">
            <Avatar className="w-24 h-24 border-2 border-namma-blue">
              <AvatarImage src={user.avatar} alt={user.name} />
              <AvatarFallback>
                <User className="h-12 w-12" />
              </AvatarFallback>
            </Avatar>
            <Button 
              variant="ghost" 
              size="sm" 
              className="mt-3 text-sm text-muted-foreground"
              onClick={() => setIsEditing(!isEditing)}
            >
              {isEditing ? "Cancel" : "Edit Profile"}
            </Button>
          </div>
          
          <div className="flex-1">
            <h2 className="text-2xl font-semibold">{user.name}</h2>
            
            <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="flex items-center text-sm">
                <Phone className="h-4 w-4 mr-2 text-muted-foreground" />
                <span>{user.phone}</span>
              </div>
              <div className="flex items-center text-sm">
                <AtSign className="h-4 w-4 mr-2 text-muted-foreground" />
                <span>{user.email}</span>
              </div>
              <div className="flex items-center text-sm">
                <MapPin className="h-4 w-4 mr-2 text-muted-foreground" />
                <span>{user.address}</span>
              </div>
            </div>
            
            <div className="mt-5 grid grid-cols-3 gap-3 text-center">
              <div className="bg-muted/50 p-3 rounded-lg">
                <div className="flex items-center justify-center mb-1">
                  <Bell className="h-5 w-5 text-namma-blue" />
                </div>
                <div className="text-lg font-semibold">{user.totalRides}</div>
                <div className="text-xs text-muted-foreground">Total Rides</div>
              </div>
              <div className="bg-muted/50 p-3 rounded-lg">
                <div className="flex items-center justify-center mb-1">
                  <Star className="h-5 w-5 text-namma-yellow fill-namma-yellow" />
                </div>
                <div className="text-lg font-semibold">{user.rating}</div>
                <div className="text-xs text-muted-foreground">Rating</div>
              </div>
              <div className="bg-muted/50 p-3 rounded-lg">
                <div className="flex items-center justify-center mb-1">
                  <Award className="h-5 w-5 text-namma-blue" />
                </div>
                <div className="text-lg font-semibold">{user.nammaPoints}</div>
                <div className="text-xs text-muted-foreground">Namma Points</div>
              </div>
            </div>
          </div>
        </div>
        
        <Tabs defaultValue="account" className="p-6 pt-0">
          <TabsList className="w-full grid grid-cols-4 mt-4">
            <TabsTrigger value="account">Account</TabsTrigger>
            <TabsTrigger value="payment">Payment</TabsTrigger>
            <TabsTrigger value="security">Security</TabsTrigger>
            <TabsTrigger value="nammacoins">NammaCoins</TabsTrigger>
          </TabsList>
          
          <TabsContent value="account" className="mt-6 space-y-4">
            {isEditing ? (
              <Card>
                <CardHeader>
                  <CardTitle>Edit Profile</CardTitle>
                  <CardDescription>
                    Make changes to your profile information here.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Name</Label>
                    <Input id="name" defaultValue={user.name} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="email">Email</Label>
                    <Input id="email" type="email" defaultValue={user.email} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="phone">Phone</Label>
                    <Input id="phone" defaultValue={user.phone} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="address">Address</Label>
                    <Input id="address" defaultValue={user.address} />
                  </div>
                </CardContent>
                <CardFooter>
                  <Button 
                    className="w-full bg-namma-blue hover:bg-namma-blue/90"
                    onClick={() => setIsEditing(false)}
                  >
                    Save Changes
                  </Button>
                </CardFooter>
              </Card>
            ) : (
              <div className="space-y-4">
                <div className="bg-muted/30 p-4 rounded-lg">
                  <div className="flex items-center mb-3">
                    <Bell className="h-5 w-5 text-namma-blue mr-2" />
                    <h3 className="font-semibold">Notifications</h3>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Configure how you receive notifications and updates about your rides.
                  </p>
                  <Button variant="outline" className="mt-3">
                    Manage Notifications
                  </Button>
                </div>
                
                <div className="bg-muted/30 p-4 rounded-lg">
                  <div className="flex items-center mb-3">
                    <MapPin className="h-5 w-5 text-namma-blue mr-2" />
                    <h3 className="font-semibold">Saved Locations</h3>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Add and manage your frequently visited locations for quick access.
                  </p>
                  <Button variant="outline" className="mt-3">
                    Manage Locations
                  </Button>
                </div>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="payment" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Payment Methods</CardTitle>
                <CardDescription>
                  Add or manage your payment options.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="bg-accent/50 p-4 rounded-lg flex items-start">
                  <CreditCard className="h-5 w-5 text-namma-blue mr-3 mt-1" />
                  <div>
                    <h4 className="font-medium mb-1">Add Payment Method</h4>
                    <p className="text-sm text-muted-foreground">
                      Add a new credit/debit card or UPI ID to pay for your rides.
                    </p>
                    <Button variant="outline" size="sm" className="mt-3">
                      Add Method
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="security" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Security</CardTitle>
                <CardDescription>
                  Manage your account security settings.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="bg-accent/50 p-4 rounded-lg flex items-start">
                  <Shield className="h-5 w-5 text-namma-blue mr-3 mt-1" />
                  <div>
                    <h4 className="font-medium mb-1">Two-Factor Authentication</h4>
                    <p className="text-sm text-muted-foreground">
                      Add an extra layer of security to your account by enabling two-factor authentication.
                    </p>
                    <Button variant="outline" size="sm" className="mt-3">
                      Enable 2FA
                    </Button>
                  </div>
                </div>
                
                <div className="bg-accent/50 p-4 rounded-lg flex items-start">
                  <User className="h-5 w-5 text-namma-blue mr-3 mt-1" />
                  <div>
                    <h4 className="font-medium mb-1">Change Password</h4>
                    <p className="text-sm text-muted-foreground">
                      Update your password regularly to keep your account secure.
                    </p>
                    <Button variant="outline" size="sm" className="mt-3">
                      Change Password
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="nammacoins" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Use Your NammaCoins</CardTitle>
                <CardDescription>
                  Redeem your Namma Points for exciting offers and rewards.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mb-4 p-4 bg-muted rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold">Available Points</h3>
                      <p className="text-2xl font-bold text-namma-blue">{user.nammaPoints}</p>
                    </div>
                    <Award className="h-8 w-8 text-namma-blue" />
                  </div>
                </div>

                <div className="grid gap-4">
                  {nammaCoinsOptions.map((option, index) => (
                    <div key={index} className="flex items-start p-4 border rounded-lg hover:bg-muted/50 transition-colors">
                      <div className="w-16 h-16 bg-white rounded-lg mr-4 flex items-center justify-center p-2">
                        <img src={option.image} alt={option.name} className="w-full h-full object-contain" />
                      </div>
                      <div className="flex-1">
                        <h4 className="font-semibold">{option.name}</h4>
                        <p className="text-sm text-muted-foreground">{option.description}</p>
                        <div className="mt-2 flex items-center justify-between">
                          <Button 
                            variant="outline" 
                            size="sm"
                            className="ml-auto"
                          >
                            Avail Voucher
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default UserProfile;
