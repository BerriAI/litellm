"use client";
import React, { useState, useEffect } from "react";
import { userInfoCall } from "./networking";
import { Grid, Col, Card, Text } from "@tremor/react";
import CreateKey from "./create_key_button";
import ViewKeyTable from "./view_key_table";
import ViewUserSpend from "./view_user_spend";
import EnterProxyUrl from "./enter_proxy_url";
import Navbar from "./navbar";
import { useSearchParams } from "next/navigation";
import { jwtDecode } from "jwt-decode";

const proxyBaseUrl = null;
// const proxyBaseUrl = "http://localhost:4000" // http://localhost:4000

type UserSpendData = {
  spend: number;
  max_budget?: number | null;
}

const UserDashboard = () => {
  const [data, setData] = useState<null | any[]>(null); // Keep the initialization of state here
  const [userSpendData, setUserSpendData] = useState<UserSpendData | null>(null);

  // Assuming useSearchParams() hook exists and works in your setup
  const searchParams = useSearchParams();
  const userID = searchParams.get("userID");
  const viewSpend = searchParams.get("viewSpend");

  const token = searchParams.get("token");
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [userRole, setUserRole] = useState<string | null>(null);


  function formatUserRole(userRole: string) {
    if (!userRole) {
      return "Undefined Role";
    }
  
    switch (userRole.toLowerCase()) {
      case "app_owner":
        return "App Owner";
      case "demo_app_owner":
          return "AppOwner";
      case "admin":
        return "Admin";
      case "app_user":
        return "App User";
      default:
        return "Unknown Role";
    }
  }

  // Moved useEffect inside the component and used a condition to run fetch only if the params are available
  useEffect(() => {
    if (token){
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
      }
    }
    if (userID && accessToken  && !data) {
      const fetchData = async () => {
        try {
          const response = await userInfoCall(
            accessToken,
            userID
          );
          console.log("Response:", response);
          setUserSpendData(response["user_info"])
          setData(response["keys"]); // Assuming this is the correct path to your data
        } catch (error) {
          console.error("There was an error fetching the data", error);
          // Optionally, update your UI to reflect the error state here as well
        }
      };
      fetchData();
    }
  }, [userID, token, accessToken, data]);

  if (userID == null || token == null) {

  
    // Now you can construct the full URL
    const url = proxyBaseUrl ? `${proxyBaseUrl}/sso/key/generate` : `/sso/key/generate`;
    console.log("Full URL:", url);
    window.location.href = url;

    return null;
  }
  else if (accessToken == null) {
    return null;
  }

  if (userRole == null) {
    setUserRole("App Owner")
  }
  
  return (
    <div>
      <Navbar
        userID={userID}
        userRole={userRole}
      />
      <Grid numItems={1} className="gap-0 p-10 h-[75vh] w-full">
      <Col numColSpan={1}>
        <ViewUserSpend
          userID={userID}
          userSpendData={userSpendData}
        />
        <ViewKeyTable
          userID={userID}
          accessToken={accessToken}
          data={data}
          setData={setData}
        />
        <CreateKey
          userID={userID}
          userRole={userRole}
          accessToken={accessToken}
          data={data}
          setData={setData}
        />
      </Col>
    </Grid>
    </div>

  );
};

export default UserDashboard;