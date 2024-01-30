"use client";
import React, { useState, useEffect } from "react";
import { userInfoCall } from "./networking";
import { Grid, Col, Card, Text } from "@tremor/react";
import CreateKey from "./create_key_button";
import ViewKeyTable from "./view_key_table";
import EnterProxyUrl from "./enter_proxy_url";
import { useSearchParams } from "next/navigation";

const UserDashboard = () => {
  const [data, setData] = useState<null | any[]>(null); // Keep the initialization of state here
  // Assuming useSearchParams() hook exists and works in your setup
  const searchParams = useSearchParams();
  const userID = searchParams.get("userID");
  const accessToken = searchParams.get("accessToken");
  const proxyBaseUrl = searchParams.get("proxyBaseUrl");

  // Moved useEffect inside the component and used a condition to run fetch only if the params are available
  useEffect(() => {
    if (userID && accessToken && proxyBaseUrl && !data) {
      const fetchData = async () => {
        try {
          const response = await userInfoCall(
            proxyBaseUrl,
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
  }, [userID, accessToken, proxyBaseUrl, data]);

  if (proxyBaseUrl == null) {
    return (
      <div>
        <EnterProxyUrl />
      </div>
    );
  }
  else if (userID == null || accessToken == null) {
    const baseUrl = proxyBaseUrl.endsWith('/') ? proxyBaseUrl : proxyBaseUrl + '/';
  
    // Now you can construct the full URL
    const url = `${baseUrl}sso/key/generate`;

    window.location.href = url;
    

    return null;
  }
  
  return (
    <Grid numItems={1} className="gap-0 p-10 h-[75vh] w-full">
      <Col numColSpan={1}>
        <ViewKeyTable
          userID={userID}
          accessToken={accessToken}
          proxyBaseUrl={proxyBaseUrl}
          data={data}
          setData={setData}
        />
        <CreateKey
          userID={userID}
          accessToken={accessToken}
          proxyBaseUrl={proxyBaseUrl}
          data={data}
          setData={setData}
        />
      </Col>
    </Grid>
  );
};

export default UserDashboard;
