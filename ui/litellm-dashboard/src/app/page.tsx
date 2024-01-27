import React from "react";
import Navbar from "../components/navbar";
import UserDashboard from "../components/user_dashboard";
const CreateKeyPage = () => {
  return (
    <div className="flex min-h-screen flex-col items-center">
      <Navbar />
      <UserDashboard />
    </div>
  );
};

export default CreateKeyPage;
