"use client";
import React from "react";
import { Grid, Col, Card, Text } from "@tremor/react";
import CreateKey from "./create_key_button";
import ViewKeyTable from "./view_key_table";
import { useSearchParams } from "next/navigation";

export default function UserDashboard() {
  const searchParams = useSearchParams();
  const userID = searchParams.get("userID");
  const accessToken = searchParams.get("accessToken");
  const proxyBaseUrl = searchParams.get("proxyBaseUrl");

  if (userID == null || accessToken == null || proxyBaseUrl == null) {
    return (
      <Card
        className="max-w-xs mx-auto"
        decoration="top"
        decorationColor="indigo"
      >
        <Text>Login to create/delete keys</Text>
      </Card>
    );
  }

  return (
    <Grid numItems={1} className="gap-0 p-10 h-[75vh]">
      <Col numColSpan={1}>
        <ViewKeyTable
          userID={userID}
          accessToken={accessToken}
          proxyBaseUrl={proxyBaseUrl}
        />
        <CreateKey
          userID={userID}
          accessToken={accessToken}
          proxyBaseUrl={proxyBaseUrl}
        />
      </Col>
    </Grid>
  );
}
