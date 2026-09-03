"use client";
import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";
import { jwtDecode } from "jwt-decode";
import React, { useEffect, useState } from "react";
import { fetchTeams } from "./common_components/fetch_teams";
import { KeyResponse, Team } from "./key_team_helpers/key_list";
import { effectiveSessionRole } from "@/utils/roles";
import { getProxyBaseUrl, keyInfoCall, modelAvailableCall, Organization, userGetInfoV2 } from "./networking";
import CreateKey, { CreateKeyPrefillData } from "./organisms/create_key_button";
import { VirtualKeysTable } from "./VirtualKeysPage/VirtualKeysTable";

export interface ProxySettings {
  PROXY_BASE_URL: string | null;
  PROXY_LOGOUT_URL: string | null;
  LITELLM_UI_API_DOC_BASE_URL?: string | null;
  DEFAULT_TEAM_DISABLED: boolean;
  SSO_ENABLED: boolean;
  DISABLE_EXPENSIVE_DB_QUERIES: boolean;
  NUM_SPEND_LOGS_ROWS: number;
}

export type UserInfo = {
  models: string[];
  max_budget?: number | null;
  spend: number;
};

interface UserDashboardProps {
  userID: string | null;
  userRole: string | null;
  userEmail: string | null;
  teams: Team[] | null;
  keys: any[] | null;
  setUserRole: React.Dispatch<React.SetStateAction<string>>;
  setUserEmail: React.Dispatch<React.SetStateAction<string | null>>;
  setTeams: React.Dispatch<React.SetStateAction<Team[] | null>>;
  setKeys: (keys: KeyResponse[]) => void;
  premiumUser: boolean;
  addKey: (data: any) => void;
  createClicked: boolean;
  autoOpenCreate?: boolean;
  prefillData?: CreateKeyPrefillData;
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
  premiumUser,
  addKey,
  createClicked,
  autoOpenCreate,
  prefillData,
}) => {
  const [userSpendData, setUserSpendData] = useState<UserInfo | null>(null);
  const [currentOrg] = useState<Organization | null>(null);

  const token = getCookie("token");

  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [selectedTeam] = useState<any | null>(null);

  // Clear session storage on page unload so next load fetches fresh data.
  // Note: MCP auth tokens are persistent and should not be cleared on page refresh
  // They are only cleared on logout
  useEffect(() => {
    const handleBeforeUnload = () => {
      const token = sessionStorage.getItem("token");
      sessionStorage.clear();
      if (token) {
        sessionStorage.setItem("token", token);
      }
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, []);

  // console.log(`selectedTeam: ${Object.entries(selectedTeam)}`);
  // Moved useEffect inside the component and used a condition to run fetch only if the params are available
  useEffect(() => {
    if (token) {
      const decoded = jwtDecode(token) as { [key: string]: any };
      if (decoded) {
        // cast decoded to dictionary

        // set accessToken
        setAccessToken(decoded.key);

        // check if userRole is defined
        if (decoded.user_role) {
          setUserRole(effectiveSessionRole(decoded.user_role));
        } else {
        }

        if (decoded.user_email) {
          setUserEmail(decoded.user_email);
        } else {
        }
      }
    }
    if (userID && accessToken && userRole && !userSpendData) {
      const cachedUserModels = sessionStorage.getItem("userModels" + userID);
      if (!cachedUserModels) {
        const fetchData = async () => {
          try {
            const response = await userGetInfoV2(accessToken, userID);

            setUserSpendData(response);

            sessionStorage.setItem("userSpendData" + userID, JSON.stringify(response));

            const model_available = await modelAvailableCall(accessToken, userID, userRole);
            // loop through model_info["data"] and create an array of element.model_name
            let available_model_names = model_available["data"].map((element: { id: string }) => element.id);

            sessionStorage.setItem("userModels" + userID, JSON.stringify(available_model_names));
          } catch (error: any) {
            console.error("There was an error fetching the data", error);
            if (error.message.includes("Invalid proxy server token passed")) {
              gotoLogin();
            }
            // Optionally, update your UI to reflect the error state here as well
          }
        };
        fetchData();
        fetchTeams(accessToken, userID, userRole, currentOrg, setTeams);
      }
    }
  }, [userID, token, accessToken, userRole]);

  useEffect(() => {
    // check key health - if it's invalid, redirect to login
    if (accessToken) {
      const fetchKeyInfo = async () => {
        try {
          await keyInfoCall(accessToken, [accessToken]);
        } catch (error: any) {
          if (error.message.includes("Invalid proxy server token passed")) {
            gotoLogin();
          }
        }
      };
      fetchKeyInfo();
    }
  }, [accessToken]);

  useEffect(() => {
    if (accessToken) {
      fetchTeams(accessToken, userID, userRole, currentOrg, setTeams);
    }
  }, [currentOrg]);

  function gotoLogin() {
    // Clear token cookies using the utility function
    clearTokenCookies();

    const baseUrl = getProxyBaseUrl();

    const url = baseUrl ? `${baseUrl}/sso/key/generate` : `/sso/key/generate`;

    window.location.href = url;

    return null;
  }

  if (token == null) {
    // user is not logged in as yet

    // Clear token cookies using the utility function
    gotoLogin();
    return null;
  } else {
    // Check if token is expired
    try {
      const decoded = jwtDecode(token) as { [key: string]: any };
      const expTime = decoded.exp;
      const currentTime = Math.floor(Date.now() / 1000);

      if (expTime && currentTime >= expTime) {
        gotoLogin();

        return null;
      }
    } catch (error) {
      console.error("Error decoding token:", error);
      // If there's an error decoding the token, consider it invalid
      clearTokenCookies();

      gotoLogin();

      return null;
    }

    if (accessToken == null) {
      return null;
    }
  }

  if (userID == null) {
    return <h1>User ID is not set</h1>;
  }

  if (userRole == null) {
    setUserRole("App Owner");
  }

  // Admin Viewer can view keys read-only — gate "Create Key" but render the
  // virtual-keys table the same as for Proxy Admin (read parity). Every
  // other role keeps its existing ability to create keys.
  const canCreateKey = userRole !== "Admin Viewer" && userRole !== "proxy_admin_viewer";

  return (
    <main className="h-[75vh] p-8">
      <div className="flex h-full flex-col">
        <VirtualKeysTable
          headerActions={
            canCreateKey ? (
              <CreateKey
                key={selectedTeam ? selectedTeam.team_id : null}
                team={selectedTeam as Team | null}
                teams={teams as Team[]}
                data={keys}
                addKey={addKey}
                autoOpenCreate={autoOpenCreate}
                prefillData={prefillData}
              />
            ) : undefined
          }
        />
      </div>
    </main>
  );
};

export default UserDashboard;
