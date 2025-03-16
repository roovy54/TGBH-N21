
import React from "react";
import Layout from "@/components/Layout";
import UserProfile from "@/components/UserProfile";

const Profile: React.FC = () => {
  return (
    <Layout>
      <div className="w-full">
        <div className="max-w-3xl mx-auto mb-6">
          <h1 className="text-2xl font-bold mb-2">Your Profile</h1>
          <p className="text-muted-foreground">Manage your account details and preferences</p>
        </div>
        
        <UserProfile />
      </div>
    </Layout>
  );
};

export default Profile;
