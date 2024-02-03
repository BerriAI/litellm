"use client";
import React, { useState, useEffect } from "react";
import { userInfoCall } from "./networking";
import { Grid, Col, Card, Text } from "@tremor/react";
import CreateKey from "./create_key_button";
import ViewKeyTable from "./view_key_table";
import EnterProxyUrl from "./enter_proxy_url";
import Navbar from "./navbar";
import { useSearchParams } from "next/navigation";
import { jwtDecode } from "jwt-decode";

// const proxyBaseUrl = null;
const proxyBaseUrl = "http://localhost:4000" // http://localhost:4000

const UserDashboard = () => {
  const [data, setData] = useState<null | any[]>(null); // Keep the initialization of state here
  // Assuming useSearchParams() hook exists and works in your setup
  const searchParams = useSearchParams();
  const userID = searchParams.get("userID");

  const token = searchParams.get("token");
  const [accessToken, setAccessToken] = useState<string | null>(null);


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
      }

    }
    if (userID && accessToken  && !data) {
      const fetchData = async () => {
        try {
          const response = await userInfoCall(
            accessToken,
            userID
          );
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

  
  return (
    <div>
      <Navbar
        userID={userID}
      />
      <Grid numItems={1} className="gap-0 p-10 h-[75vh] w-full">
      <Col numColSpan={1}>
        <ViewKeyTable
          userID={userID}
          accessToken={accessToken}
          data={data}
          setData={setData}
        />
        <CreateKey
          userID={userID}
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