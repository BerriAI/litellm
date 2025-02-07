"use client";
import React, { useState, useEffect } from "react";
import {
  userInfoCall,
  modelAvailableCall,
  getTotalSpendCall,
  getProxyUISettings,
  teamListCall,
} from "./networking";
import { Grid, Col, Card, Text, Title } from "@tremor/react";
import CreateKey from "./create_key_button";
import ViewKeyTable from "./view_key_table";
import ViewUserSpend from "./view_user_spend";
import ViewUserTeam from "./view_user_team";
import DashboardTeam from "./dashboard_default_team";
import Onboarding from "../app/onboarding/page";
import { useSearchParams, useRouter } from "next/navigation";
import { jwtDecode } from "jwt-decode";
import { Typography } from "antd";
const isLocal = process.env.NODE_ENV === "development";
if (isLocal != true) {
  console.log = function() {};
}
console.log("isLocal:", isLocal);
const proxyBaseUrl = isLocal ? "http://localhost:4000" : null;

export interface ProxySettings {
  PROXY_BASE_URL: string | null;
  PROXY_LOGOUT_URL: string | null;
  DEFAULT_TEAM_DISABLED: boolean;
  SSO_ENABLED: boolean;
  DISABLE_EXPENSIVE_DB_QUERIES: boolean;
  NUM_SPEND_LOGS_ROWS: number;
}


export type UserInfo = {
  models: string[];
  max_budget?: number | null;
  spend: number;
}

function getCookie(name: string) {
  console.log("COOKIES", document.cookie)
  const cookieValue = document.cookie
      .split('; ')
      .find(row => row.startsWith(name + '='));
  return cookieValue ? cookieValue.split('=')[1] : null;
}

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
  premiumUser: boolean;
}

type TeamInterface = {
  models: any[];
  team_id: null;
  team_alias: String;
};

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
  premiumUser,
}) => {
  const [userSpendData, setUserSpendData] = useState<UserInfo | null>(
    null
  );

  // Assuming useSearchParams() hook exists and works in your setup
  const searchParams = useSearchParams()!;
  const viewSpend = searchParams.get("viewSpend");
  const router = useRouter();

  const token = getCookie('token');

  const invitation_id = searchParams.get("invitation_id");

  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [teamSpend, setTeamSpend] = useState<number | null>(null);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [proxySettings, setProxySettings] = useState<ProxySettings | null>(null);
  const defaultTeam: TeamInterface = {
    models: [],
    team_alias: "Default Team",
    team_id: null,
  };
  const [selectedTeam, setSelectedTeam] = useState<any | null>(
    teams ? teams[0] : defaultTeam
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
      case "internal_user":
        return "Internal User";
      case "internal_user_viewer":
        return "Internal Viewer";
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
        const fetchTeams = async () => {
          let givenTeams;
          if (userRole != "Admin" && userRole != "Admin Viewer") {
            givenTeams = await teamListCall(accessToken, userID)
          } else {
            givenTeams = await teamListCall(accessToken)
          }
          
          console.log(`givenTeams: ${givenTeams}`)

          setTeams(givenTeams)
        }
        const fetchData = async () => {
          try {
            const proxy_settings: ProxySettings = await getProxyUISettings(accessToken);
            setProxySettings(proxy_settings);

            const response = await userInfoCall(
              accessToken,
              userID,
              userRole,
              false,
              null,
              null
            );
            console.log(
              `received teams in user dashboard: ${Object.keys(
                response
              )}; team values: ${Object.entries(response.teams)}`
            );

            setUserSpendData(response["user_info"]);
            console.log(`userSpendData: ${JSON.stringify(userSpendData)}`)

            // set keys for admin and users
            if (!response?.teams[0].keys) {
              setKeys(response["keys"]); 
            } else {
              setKeys(
                response["keys"].concat(
                  response.teams
                    .filter((team: any) => userRole === "Admin" || team.user_id === userID)
                    .flatMap((team: any) => team.keys)
                )
              );
              
            }

            const teamsArray = [...response["teams"]];
            if (teamsArray.length > 0) {
              console.log(`response['teams']: ${JSON.stringify(teamsArray)}`);
              setSelectedTeam(teamsArray[0]);
            } else {
              setSelectedTeam(defaultTeam);
              
            }
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
        fetchTeams();
      }
    }
  }, [userID, token, accessToken, keys, userRole]);

  useEffect(() => {
    // This code will run every time selectedTeam changes
    if (
      keys !== null &&
      selectedTeam !== null &&
      selectedTeam !== undefined &&
      selectedTeam.team_id !== null
    ) {
      let sum = 0;
      console.log(`keys: ${JSON.stringify(keys)}`)
      for (const key of keys) {
        if (
          selectedTeam.hasOwnProperty("team_id") &&
          key.team_id !== null &&
          key.team_id === selectedTeam.team_id
        ) {
          sum += key.spend;
        }
      }
      console.log(`sum: ${sum}`)
      setTeamSpend(sum);
    } else if (keys !== null) {
      // sum the keys which don't have team-id set (default team)
      let sum = 0;
      for (const key of keys) {
        sum += key.spend;
      }
      setTeamSpend(sum);
    }
  }, [selectedTeam]);


  if (invitation_id != null) {
    return (
      <Onboarding></Onboarding>
    )
  }

  if (userID == null || token == null) {
    // user is not logged in as yet 
    const url = proxyBaseUrl
      ? `${proxyBaseUrl}/sso/key/generate`
      : `/sso/key/generate`;
    

    // clear cookie called "token" since user will be logging in again
    document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";

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

  console.log("inside user dashboard, selected team", selectedTeam);
  return (
    <div className="w-full mx-4">
      <Grid numItems={1} className="gap-2 p-8 h-[75vh] w-full mt-2">
        <Col numColSpan={1}>
          <ViewUserTeam
            userID={userID}
            userRole={userRole}
            selectedTeam={selectedTeam ? selectedTeam : null}
            accessToken={accessToken}
          />
          <ViewUserSpend
            userID={userID}
            userRole={userRole}
            userMaxBudget={userSpendData?.max_budget || null}
            accessToken={accessToken}
            userSpend={teamSpend}
            selectedTeam={selectedTeam ? selectedTeam : null}
          />

          <ViewKeyTable
            userID={userID}
            userRole={userRole}
            accessToken={accessToken}
            selectedTeam={selectedTeam ? selectedTeam : null}
            data={keys}
            setData={setKeys}
            premiumUser={premiumUser}
            teams={teams}
          />
          <CreateKey
            key={selectedTeam ? selectedTeam.team_id : null}
            userID={userID}
            team={selectedTeam ? selectedTeam : null}
            userRole={userRole}
            accessToken={accessToken}
            data={keys}
            setData={setKeys}
          />
          <DashboardTeam
            teams={teams}
            setSelectedTeam={setSelectedTeam}
            userRole={userRole}
            proxySettings={proxySettings}
            setProxySettings={setProxySettings}
            userInfo={userSpendData}
            accessToken={accessToken}
            setKeys={setKeys}
          />
        </Col>
      </Grid>
    </div>
  );
};

export default UserDashboard;
