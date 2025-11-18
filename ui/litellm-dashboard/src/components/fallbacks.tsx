import { PlayIcon, TrashIcon } from "@heroicons/react/outline";
import { Icon, Table, TableBody, TableCell, TableHead, TableHeaderCell, TableRow } from "@tremor/react";
import { Modal, Tooltip } from "antd";
import openai from "openai";
import React, { useEffect, useState } from "react";
import AddFallbacks from "./add_fallbacks";
import NotificationsManager from "./molecules/notifications_manager";
import { getCallbacksCall, setCallbacksCall } from "./networking";

type FallbackEntry = { [modelName: string]: string[] };
type Fallbacks = FallbackEntry[];

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
  const proxyBaseUrl = isLocal ? "http://localhost:4000" : window.location.origin;
  const client = new openai.OpenAI({
    apiKey: accessToken,
    baseURL: proxyBaseUrl,
    dangerouslyAllowBrowser: true,
  });

  try {
    NotificationsManager.info("Testing fallback model response...");

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
  const [isDeleting, setIsDeleting] = useState(false);
  const [fallbackToDelete, setFallbackToDelete] = useState<FallbackEntry | null>(null);

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

  const handleDeleteClick = (fallbackEntry: FallbackEntry) => {
    setFallbackToDelete(fallbackEntry);
  };

  const handleDeleteConfirm = async () => {
    if (!fallbackToDelete || !accessToken) {
      return;
    }

    const key = Object.keys(fallbackToDelete)[0];
    if (!key) {
      return;
    }
    setIsDeleting(true);

    const updatedFallbacks = routerSettings["fallbacks"]
      .map((dict: FallbackEntry) => {
        if (key in dict && Array.isArray(dict[key])) {
          delete dict[key];
        }
        return dict;
      })
      .filter((dict: FallbackEntry) => Object.keys(dict).length > 0);

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
    } finally {
      setIsDeleting(false);
      setFallbackToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setFallbackToDelete(null);
  };

  if (!accessToken) {
    return null;
  }

  return (
    <>
      <AddFallbacks
        models={modelData?.data ? modelData.data.map((data: any) => data.model_name) : []}
        accessToken={accessToken}
        routerSettings={routerSettings}
        setRouterSettings={setRouterSettings}
      />
      <Table>
        <TableHead>
          <TableRow>
            <TableHeaderCell>Model Name</TableHeaderCell>
            <TableHeaderCell>Fallbacks</TableHeaderCell>
            <TableHeaderCell>Actions</TableHeaderCell>
          </TableRow>
        </TableHead>

        <TableBody>
          {routerSettings["fallbacks"] &&
            routerSettings["fallbacks"].map((item: FallbackEntry, index: number) =>
              Object.entries(item).map(([key, value]) => (
                <TableRow key={index.toString() + key}>
                  <TableCell>{key}</TableCell>
                  <TableCell>{Array.isArray(value) ? value.join(", ") : value}</TableCell>
                  <TableCell>
                    <Tooltip title="Test fallback">
                      <Icon
                        icon={PlayIcon}
                        size="sm"
                        onClick={() => testFallbackModelResponse(Object.keys(item)[0], accessToken || "")}
                        className="cursor-pointer hover:text-blue-600"
                      />
                    </Tooltip>
                    <Tooltip title="Delete fallback">
                      <Icon
                        icon={TrashIcon}
                        size="sm"
                        onClick={() => handleDeleteClick(item)}
                        className="cursor-pointer hover:text-red-600"
                      />
                    </Tooltip>
                  </TableCell>
                </TableRow>
              )),
            )}
        </TableBody>
      </Table>
      {fallbackToDelete && (
        <Modal
          title="Delete Fallback"
          open={fallbackToDelete !== null}
          onOk={handleDeleteConfirm}
          onCancel={handleDeleteCancel}
          confirmLoading={isDeleting}
          okText="Delete"
          okButtonProps={{ danger: true }}
        >
          <p>Are you sure you want to delete fallback: {Object.keys(fallbackToDelete)[0]} ?</p>
          <p>This action cannot be undone.</p>
        </Modal>
      )}
    </>
  );
};

export default Fallbacks;
