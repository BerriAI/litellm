import React, { useState, useEffect } from "react";
import {
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableCell,
  TableBody,
  Button,
  Icon,
} from "@tremor/react";
import {
  getCallbacksCall,
  setCallbacksCall,
} from "./networking";
import { TrashIcon } from "@heroicons/react/outline";
import AddFallbacks from "./add_fallbacks";
import openai from "openai";
import NotificationsManager from "./molecules/notifications_manager";

interface FallbacksProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  modelData: any;
}

async function testFallbackModelResponse(selectedModel: string, accessToken: string) {
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal != true) {
    console.log = function () {};
  }
  console.log("isLocal:", isLocal);
  const proxyBaseUrl = isLocal ? "http://localhost:4000" : window.location.origin;
  const client = new openai.OpenAI({
    apiKey: accessToken,
    baseURL: proxyBaseUrl,
    dangerouslyAllowBrowser: true,
  });

  try {
    const response = await client.chat.completions.create({
      model: selectedModel,
      messages: [
        {
          role: "user",
          content: "Hi, this is a test message",
        },
      ],
      // @ts-ignore
      mock_testing_fallbacks: true,
    });

    NotificationsManager.success(
      <span>
        Test model=<strong>{selectedModel}</strong>, received model=
        <strong>{response.model}</strong>. See{" "}
        <a
          href="#"
          onClick={() => window.open("https://docs.litellm.ai/docs/proxy/reliability", "_blank")}
          style={{ textDecoration: "underline", color: "blue" }}
        >
          curl
        </a>
      </span>,
    );
  } catch (error) {
    NotificationsManager.fromBackend(
      `Error occurred while generating model response. Please try again. Error: ${error}`,
    );
  }
}

const Fallbacks: React.FC<FallbacksProps> = ({ accessToken, userRole, userID, modelData }) => {
  const [routerSettings, setRouterSettings] = useState<{ [key: string]: any }>({});

  useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
    getCallbacksCall(accessToken, userID, userRole).then((data) => {
      console.log("callbacks", data);
      let router_settings = data.router_settings;
      if ("model_group_retry_policy" in router_settings) {
        delete router_settings["model_group_retry_policy"];
      }
      setRouterSettings(router_settings);
    });
  }, [accessToken, userRole, userID]);

  const deleteFallbacks = async (key: string) => {
    if (!accessToken) {
      return;
    }

    console.log(`received key: ${key}`);
    console.log(`routerSettings['fallbacks']: ${routerSettings["fallbacks"]}`);

    const updatedFallbacks = routerSettings["fallbacks"]
      .map((dict: { [key: string]: any }) => {
        if (key in dict) {
          delete dict[key];
        }
        return dict;
      })
      .filter((dict: { [key: string]: any }) => Object.keys(dict).length > 0);

    const updatedSettings = {
      ...routerSettings,
      fallbacks: updatedFallbacks,
    };

    const payload = {
      router_settings: updatedSettings,
    };

    try {
      await setCallbacksCall(accessToken, payload);
      setRouterSettings(updatedSettings);
      NotificationsManager.success("Router settings updated successfully");
    } catch (error) {
      NotificationsManager.fromBackend("Failed to update router settings: " + error);
    }
  };

  if (!accessToken) {
    return null;
  }

  return (
    <>
      <Table>
        <TableHead>
          <TableRow>
            <TableHeaderCell>Model Name</TableHeaderCell>
            <TableHeaderCell>Fallbacks</TableHeaderCell>
          </TableRow>
        </TableHead>

        <TableBody>
          {routerSettings["fallbacks"] &&
            routerSettings["fallbacks"].map((item: object, index: number) =>
              Object.entries(item).map(([key, value]) => (
                <TableRow key={index.toString() + key}>
                  <TableCell>{key}</TableCell>
                  <TableCell>{Array.isArray(value) ? value.join(", ") : value}</TableCell>
                  <TableCell>
                    <Button onClick={() => testFallbackModelResponse(key, accessToken)}>Test Fallback</Button>
                  </TableCell>
                  <TableCell>
                    <Icon icon={TrashIcon} size="sm" onClick={() => deleteFallbacks(key)} />
                  </TableCell>
                </TableRow>
              )),
            )}
        </TableBody>
      </Table>
      <AddFallbacks
        models={modelData?.data ? modelData.data.map((data: any) => data.model_name) : []}
        accessToken={accessToken}
        routerSettings={routerSettings}
        setRouterSettings={setRouterSettings}
      />
    </>
  );
};

export default Fallbacks;

