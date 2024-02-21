"use client";
import React, { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Navbar from "../components/navbar";
import UserDashboard from "../components/user_dashboard";
import ModelDashboard from "@/components/model_dashboard";
import ViewUserDashboard from "@/components/view_users";
import ChatUI from "@/components/chat_ui";
import Sidebar from "../components/leftnav";
import Usage from "../components/usage";
import { jwtDecode } from "jwt-decode";

const CreateKeyPage = () => {
  const [userRole, setUserRole] = useState('');
  const [userEmail, setUserEmail] = useState<null | string>(null);
  const searchParams = useSearchParams();

  const userID = searchParams.get("userID");
  const token = searchParams.get("token");

  const [page, setPage] = useState("api-keys");
  const [accessToken, setAccessToken] = useState<string | null>(null);

  useEffect(() => {
    if (token) {
      const decoded = jwtDecode(token) as { [key: string]: any };
      if (decoded) {
        // cast decoded to dictionary
        console.log("Decoded token:", decoded);

        console.log("Decoded key:", decoded.key);
        // set accessToken
        setAccessToken(decoded.key);

        // check if userRole is defined
        if (decoded.user_role) {
          const formattedUserRole = formatUserRole(decoded.user_role);
          console.log("Decoded user_role:", formattedUserRole);
          setUserRole(formattedUserRole);
        } else {
          console.log("User role not defined");
        }

        if (decoded.user_email) {
          setUserEmail(decoded.user_email);
        } else {
          console.log(`User Email is not set ${decoded}`);
        }
      }
    }
  }, [token]);

  function formatUserRole(userRole: string) {
    if (!userRole) {
      return "Undefined Role";
    }
    console.log(`Received user role: ${userRole}`);
    switch (userRole.toLowerCase()) {
      case "app_owner":
        return "App Owner";
      case "demo_app_owner":
        return "App Owner";
      case "app_admin":
        return "Admin";
      case "app_user":
        return "App User";
      default:
        return "Unknown Role";
    }
  }

  return (
    <Suspense fallback={<div>Loading...</div>}>
      <div className="flex flex-col min-h-screen">
        <Navbar userID={userID} userRole={userRole} userEmail={userEmail} />
        <div className="flex flex-1 overflow-auto">
          <Sidebar setPage={setPage} userRole={userRole}/>
          {page == "api-keys" ? (
            <UserDashboard
              userID={userID}
              userRole={userRole}
              setUserRole={setUserRole}
              userEmail={userEmail}
              setUserEmail={setUserEmail}
            />
          ) : page == "models" ? (
            <ModelDashboard
              userID={userID}
              userRole={userRole}
              token={token}
              accessToken={accessToken}
            />
          ) : page == "llm-playground" ? (
            <ChatUI
              userID={userID}
              userRole={userRole}
              token={token}
              accessToken={accessToken}
            />
          )
          : page == "users" ? (
            <ViewUserDashboard
              userID={userID}
              userRole={userRole}
              token={token}
              accessToken={accessToken}
            />
          )
          : (
            <Usage
              userID={userID}
              userRole={userRole}
              token={token}
              accessToken={accessToken}
            />
          )}
        </div>
      </div>
    </Suspense>
  );
};

export default CreateKeyPage;
