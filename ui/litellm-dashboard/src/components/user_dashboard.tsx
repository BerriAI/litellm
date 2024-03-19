"use client";
import React, { useState, useEffect } from "react";
import { userInfoCall, modelAvailableCall } from "./networking";
import { Grid, Col, Card, Text } from "@tremor/react";
import CreateKey from "./create_key_button";
import ViewKeyTable from "./view_key_table";
import ViewUserSpend from "./view_user_spend";
import DashboardTeam from "./dashboard_default_team";
import { useSearchParams, useRouter } from "next/navigation";
import { jwtDecode } from "jwt-decode";
import { Typography } from "antd";
const isLocal = process.env.NODE_ENV === "development";
console.log("isLocal:", isLocal);
const proxyBaseUrl = isLocal ? "http://localhost:4000" : null;

type UserSpendData = {
  spend: number;
  max_budget?: number | null;
};

interface UserDashboardProps {
  userID: string | null;
  userRole: string | null;
  userEmail: string | null;
  teams: any[] | null;
  keys: any[] | null;
  setUserRole: React.Dispatch<React.SetStateAction<string>>;
  setUserEmail: React.Dispatch<React.SetStateAction<string | null>>;
  setTeams: React.Dispatch<React.SetStateAction<Object[] | null>>;
  setKeys: React.Dispatch<React.SetStateAction<Object[] | null>>;
}

const UserDashboard: React.FC<UserDashboardProps> = ({
  userID,
  userRole,
  teams,
  keys,
  setUserRole,
  userEmail,
  setUserEmail,
  setTeams,
  setKeys,
}) => {
  const [userSpendData, setUserSpendData] = useState<UserSpendData | null>(
    null
  );

  // Assuming useSearchParams() hook exists and works in your setup
  const searchParams = useSearchParams();
  const viewSpend = searchParams.get("viewSpend");
  const router = useRouter();

  const token = searchParams.get("token");
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [selectedTeam, setSelectedTeam] = useState<any | null>(
    teams ? teams[0] : null
  );
  // check if window is not undefined
  if (typeof window !== "undefined") {
    window.addEventListener("beforeunload", function () {
      // Clear session storage
      sessionStorage.clear();
    });
  }

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
      case "proxy_admin":
        return "Admin";
      case "proxy_admin_viewer":
        return "Admin Viewer";
      case "app_user":
        return "App User";
      default:
        return "Unknown Role";
    }
  }

  // console.log(`selectedTeam: ${Object.entries(selectedTeam)}`);
  // Moved useEffect inside the component and used a condition to run fetch only if the params are available
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
    if (userID && accessToken && userRole && !keys && !userSpendData) {
      const cachedUserModels = sessionStorage.getItem("userModels" + userID);
      if (cachedUserModels) {
        setUserModels(JSON.parse(cachedUserModels));
      } else {
        const fetchData = async () => {
          try {
            const response = await userInfoCall(accessToken, userID, userRole);
            console.log(
              `received teams in user dashboard: ${Object.keys(
                response
              )}; team values: ${Object.entries(response.teams)}`
            );
            setUserSpendData(response["user_info"]);
            setKeys(response["keys"]); // Assuming this is the correct path to your data
            setTeams(response["teams"]);
            setSelectedTeam(response["teams"] ? response["teams"][0] : null);
            sessionStorage.setItem(
              "userData" + userID,
              JSON.stringify(response["keys"])
            );
            sessionStorage.setItem(
              "userSpendData" + userID,
              JSON.stringify(response["user_info"])
            );

            const model_available = await modelAvailableCall(
              accessToken,
              userID,
              userRole
            );
            // loop through model_info["data"] and create an array of element.model_name
            let available_model_names = model_available["data"].map(
              (element: { id: string }) => element.id
            );
            console.log("available_model_names:", available_model_names);
            setUserModels(available_model_names);

            console.log("userModels:", userModels);

            sessionStorage.setItem(
              "userModels" + userID,
              JSON.stringify(available_model_names)
            );
          } catch (error) {
            console.error("There was an error fetching the data", error);
            // Optionally, update your UI to reflect the error state here as well
          }
        };
        fetchData();
      }
    }
  }, [userID, token, accessToken, keys, userRole]);

  if (userID == null || token == null) {
    // Now you can construct the full URL
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/sso/key/generate`
      : `/sso/key/generate`;
    console.log("Full URL:", url);
    window.location.href = url;

    return null;
  } else if (accessToken == null) {
    return null;
  }

  if (userRole == null) {
    setUserRole("App Owner");
  }

  if (userRole && userRole == "Admin Viewer") {
    const { Title, Paragraph } = Typography;
    return (
      <div>
        <Title level={1}>Access Denied</Title>
        <Paragraph>Ask your proxy admin for access to create keys</Paragraph>
      </div>
    );
  }

  return (
    <div>
      <Grid numItems={1} className="gap-0 p-10 h-[75vh] w-full">
        <Col numColSpan={1}>
          <ViewUserSpend
            userID={userID}
            userSpendData={userSpendData}
            userRole={userRole}
            accessToken={accessToken}
          />
          <ViewKeyTable
            userID={userID}
            accessToken={accessToken}
            data={keys}
            setData={setKeys}
          />
          <CreateKey
            userID={userID}
            teamID={selectedTeam ? selectedTeam["team_id"] : null}
            userRole={userRole}
            userModels={userModels}
            accessToken={accessToken}
            data={keys}
            setData={setKeys}
          />
          <DashboardTeam teams={teams} setSelectedTeam={setSelectedTeam} />
        </Col>
      </Grid>
    </div>
  );
};

export default UserDashboard;
