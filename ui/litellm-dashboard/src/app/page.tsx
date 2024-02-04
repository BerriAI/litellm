import React, { Suspense } from "react";
import Navbar from "../components/navbar";
import UserDashboard from "../components/user_dashboard";
const CreateKeyPage = () => {
  return (
    <Suspense fallback={<div>Loading...</div>}>
    <div className="flex min-h-screen flex-col ">
      <UserDashboard />
    </div>
    </Suspense>
  );
};

export default CreateKeyPage;
