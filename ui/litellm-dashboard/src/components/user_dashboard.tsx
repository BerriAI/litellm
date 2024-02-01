"use client";
import React, { useState, useEffect } from "react";
import { userInfoCall } from "./networking";
import { Grid, Col, Card, Text } from "@tremor/react";
import CreateKey from "./create_key_button";
import ViewKeyTable from "./view_key_table";
import EnterProxyUrl from "./enter_proxy_url";
import { useSearchParams } from "next/navigation";
import { jwtDecode } from "jwt-decode";

const UserDashboard = () => {
  const [data, setData] = useState<null | any[]>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const searchParams = useSearchParams();
  const userID = searchParams.get("userID");
  const proxyBaseUrl = searchParams.get("proxyBaseUrl");
  const token = searchParams.get("token");

  useEffect(() => {
    const fetchData = async () => {
      try {
        if (token) {
          const decoded = jwtDecode(token) as { key: string };
          console.log("Decoded token:", decoded);
          console.log("Decoded key:", decoded.key);
          setAccessToken(decoded.key);
        }

        if (userID && accessToken && proxyBaseUrl && !data) {
          const response = await userInfoCall(proxyBaseUrl, accessToken, userID);
          setData(response["keys"]);
        }
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false); // Set loading to false when the effect is done
      }
    };

    fetchData();
  }, [token, userID, accessToken, proxyBaseUrl, data]);

  if (loading) {
    return <div>Loading...</div>;
  }

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
