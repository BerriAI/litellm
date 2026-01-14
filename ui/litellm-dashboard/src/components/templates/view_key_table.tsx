"use client";
import React, { useEffect, useState } from "react";
import { keyDeleteCall, Organization } from "../networking";
import { add } from "date-fns";
import { regenerateKeyCall } from "../networking";
import { Grid, Col, Button, Text, Title, TextInput } from "@tremor/react";
import { fetchAvailableModelsForTeamOrKey } from "../key_team_helpers/fetch_available_models_team_key";
import { Modal, Form, InputNumber } from "antd";
import { CopyToClipboard } from "react-copy-to-clipboard";
import useKeyList from "../key_team_helpers/key_list";
import { KeyResponse } from "../key_team_helpers/key_list";
import { AllKeysTable } from "../all_keys_table";
import { Team } from "../key_team_helpers/key_list";
import { Setter } from "@/types";

import NotificationManager from "../molecules/notifications_manager";

interface EditKeyModalProps {
  visible: boolean;
  onCancel: () => void;
  token: any; // Assuming TeamType is a type representing your team object
  onSubmit: (data: FormData) => void; // Assuming FormData is the type of data to be submitted
}

interface ModelLimitModalProps {
  visible: boolean;
  onCancel: () => void;
  token: KeyResponse;
  onSubmit: (updatedMetadata: any) => void;
  accessToken: string;
}

// Define the props type
interface ViewKeyTableProps {
  userID: string | null;
  userRole: string | null;
  accessToken: string | null;
  selectedTeam: Team | null;
  setSelectedTeam: React.Dispatch<React.SetStateAction<any | null>>;
  data: KeyResponse[] | null;
  setData: (keys: KeyResponse[]) => void;
  teams: Team[] | null;
  premiumUser: boolean;
  currentOrg: Organization | null;
  organizations: Organization[] | null;
  setCurrentOrg: React.Dispatch<React.SetStateAction<Organization | null>>;
  selectedKeyAlias: string | null;
  setSelectedKeyAlias: Setter<string | null>;
  createClicked: boolean;
  setAccessToken?: (token: string) => void;
}

interface ItemData {
  key_alias: string | null;
  key_name: string;
  spend: string;
  max_budget: string | null;
  models: string[];
  tpm_limit: string | null;
  rpm_limit: string | null;
  token: string;
  token_id: string | null;
  id: number;
  team_id: string;
  metadata: any;
  user_id: string | null;
  expires: any;
  budget_duration: string | null;
  budget_reset_at: string | null;
  // Add any other properties that exist in the item data
}

interface ModelLimits {
  [key: string]: number; // Index signature allowing string keys
}

interface CombinedLimit {
  tpm: number;
  rpm: number;
}

interface CombinedLimits {
  [key: string]: CombinedLimit; // Index signature allowing string keys
}

const ViewKeyTable: React.FC<ViewKeyTableProps> = ({
  userID,
  userRole,
  accessToken,
  selectedTeam,
  setSelectedTeam,
  data,
  setData,
  teams,
  premiumUser,
  currentOrg,
  organizations,
  setCurrentOrg,
  selectedKeyAlias,
  setSelectedKeyAlias,
  createClicked,
  setAccessToken,
}) => {
  const [isButtonClicked, setIsButtonClicked] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [keyToDelete, setKeyToDelete] = useState<string | null>(null);
  const [deleteConfirmInput, setDeleteConfirmInput] = useState("");
  const [selectedItem, setSelectedItem] = useState<KeyResponse | null>(null);
  const [spendData, setSpendData] = useState<{ day: string; spend: number }[] | null>(null);

  // NEW: Declare filter states for team and key alias.
  const [teamFilter, setTeamFilter] = useState<string>(selectedTeam?.team_id || "");

  // Keep the team filter in sync with the incoming prop.
  useEffect(() => {
    setTeamFilter(selectedTeam?.team_id || "");
  }, [selectedTeam]);

  // Build a memoized filters object for the backend call.

  // Pass filters into the hook so the API call includes these query parameters.
  const { keys, isLoading, error, pagination, refresh, setKeys } = useKeyList({
    selectedTeam: selectedTeam || undefined,
    currentOrg,
    selectedKeyAlias,
    accessToken: accessToken || "",
    createClicked,
  });

  const handlePageChange = (newPage: number) => {
    refresh({ page: newPage });
  };

  const [editModalVisible, setEditModalVisible] = useState(false);
  const [infoDialogVisible, setInfoDialogVisible] = useState(false);
  const [selectedToken, setSelectedToken] = useState<KeyResponse | null>(null);
  const [userModels, setUserModels] = useState<string[]>([]);
  const initialKnownTeamIDs: Set<string> = new Set();
  const [modelLimitModalVisible, setModelLimitModalVisible] = useState(false);
  const [regenerateDialogVisible, setRegenerateDialogVisible] = useState(false);
  const [regeneratedKey, setRegeneratedKey] = useState<string | null>(null);
  const [regenerateFormData, setRegenerateFormData] = useState<any>(null);
  const [regenerateForm] = Form.useForm();
  const [newExpiryTime, setNewExpiryTime] = useState<string | null>(null);

  const [knownTeamIDs, setKnownTeamIDs] = useState(initialKnownTeamIDs);
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);

  useEffect(() => {
    const calculateNewExpiryTime = (duration: string | undefined) => {
      if (!duration) {
        return null;
      }

      try {
        const now = new Date();
        let newExpiry: Date;

        if (duration.endsWith("s")) {
          newExpiry = add(now, { seconds: parseInt(duration) });
        } else if (duration.endsWith("h")) {
          newExpiry = add(now, { hours: parseInt(duration) });
        } else if (duration.endsWith("d")) {
          newExpiry = add(now, { days: parseInt(duration) });
        } else {
          throw new Error("Invalid duration format");
        }

        return newExpiry.toLocaleString("en-US", {
          year: "numeric",
          month: "numeric",
          day: "numeric",
          hour: "numeric",
          minute: "numeric",
          second: "numeric",
          hour12: true,
        });
      } catch (error) {
        return null;
      }
    };

    console.log("in calculateNewExpiryTime for selectedToken", selectedToken);

    // When a new duration is entered
    if (regenerateFormData?.duration) {
      setNewExpiryTime(calculateNewExpiryTime(regenerateFormData.duration));
    } else {
      setNewExpiryTime(null);
    }

    console.log("calculateNewExpiryTime:", newExpiryTime);
  }, [selectedToken, regenerateFormData?.duration]);

  useEffect(() => {
    const fetchUserModels = async () => {
      try {
        if (userID === null || userRole === null || accessToken === null) {
          return;
        }

        const models = await fetchAvailableModelsForTeamOrKey(userID, userRole, accessToken);
        if (models) {
          setUserModels(models);
        }
      } catch (error) {
        NotificationManager.error({ description: "Error fetching user models" });
      }
    };

    fetchUserModels();
  }, [accessToken, userID, userRole]);

  useEffect(() => {
    if (teams) {
      const teamIDSet: Set<string> = new Set();
      teams.forEach((team: any, index: number) => {
        const team_obj: string = team.team_id;
        teamIDSet.add(team_obj);
      });
      setKnownTeamIDs(teamIDSet);
    }
  }, [teams]);

  const confirmDelete = async () => {
    if (keyToDelete == null || keys == null) {
      return;
    }

    try {
      if (!accessToken) return;
      await keyDeleteCall(accessToken, keyToDelete);
      // Successfully completed the deletion. Update the state to trigger a rerender.
      const filteredKeys = keys.filter((item) => item.token !== keyToDelete);
      setKeys(filteredKeys);
    } catch (error) {
      NotificationManager.error({ description: "Error deleting the key" });
    }

    // Close the confirmation modal and reset the keyToDelete
    setIsDeleteModalOpen(false);
    setKeyToDelete(null);
    setDeleteConfirmInput("");
  };

  const cancelDelete = () => {
    // Close the confirmation modal and reset the keyToDelete
    setIsDeleteModalOpen(false);
    setKeyToDelete(null);
    setDeleteConfirmInput("");
  };

  const handleRegenerateClick = (token: any) => {
    setSelectedToken(token);
    setNewExpiryTime(null);
    regenerateForm.setFieldsValue({
      key_alias: token.key_alias,
      max_budget: token.max_budget,
      tpm_limit: token.tpm_limit,
      rpm_limit: token.rpm_limit,
      duration: token.duration || "",
    });
    setRegenerateDialogVisible(true);
  };

  const handleRegenerateFormChange = (field: string, value: any) => {
    setRegenerateFormData((prev: any) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleRegenerateKey = async () => {
    if (!premiumUser) {
      NotificationManager.warning({
        description: "Regenerate Virtual Key is an Enterprise feature. Please upgrade to use this feature.",
      });
      return;
    }

    if (selectedToken == null) {
      return;
    }

    try {
      const formValues = await regenerateForm.validateFields();
      if (!accessToken) return;
      const response = await regenerateKeyCall(accessToken, selectedToken.token || selectedToken.token_id, formValues);
      setRegeneratedKey(response.key);

      // Update the data state with the new key_name
      if (data) {
        const updatedData = data.map((item) =>
          item.token === selectedToken?.token ? { ...item, key_name: response.key_name, ...formValues } : item,
        );
        setData(updatedData);
      }

      setRegenerateDialogVisible(false);
      regenerateForm.resetFields();
      NotificationManager.success({ description: "Virtual Key regenerated successfully" });
    } catch (error) {
      console.error("Error regenerating key:", error);
      NotificationManager.error({ description: "Failed to regenerate Virtual Key" });
    }
  };

  return (
    <div>
      <AllKeysTable
        keys={keys}
        setKeys={setKeys}
        isLoading={isLoading}
        pagination={pagination}
        onPageChange={handlePageChange}
        pageSize={100}
        teams={teams}
        selectedTeam={selectedTeam}
        setSelectedTeam={setSelectedTeam}
        accessToken={accessToken}
        userID={userID}
        userRole={userRole}
        organizations={organizations}
        setCurrentOrg={setCurrentOrg}
        refresh={refresh}
        selectedKeyAlias={selectedKeyAlias}
        setSelectedKeyAlias={setSelectedKeyAlias}
        premiumUser={premiumUser}
        setAccessToken={setAccessToken}
      />

      {isDeleteModalOpen &&
        (() => {
          const keyData = keys?.find((k) => k.token === keyToDelete);
          const keyName = keyData?.key_alias || keyData?.token_id || keyToDelete;
          const isValid = deleteConfirmInput === keyName;
          return (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl min-h-[380px] py-6 overflow-hidden transform transition-all flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-900">Delete Key</h3>
                    <button
                      onClick={() => {
                        cancelDelete();
                        setDeleteConfirmInput("");
                      }}
                      className="text-gray-400 hover:text-gray-500 focus:outline-none"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                  <div className="px-6 py-4">
                    <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-100 rounded-md mb-5">
                      <div className="text-red-500 mt-0.5">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.082 16.5c-.77.833.192 2.5 1.732 2.5z"
                          />
                        </svg>
                      </div>
                      <div>
                        <p className="text-base font-medium text-red-600">
                          Warning: You are about to delete this Virtual Key.
                        </p>
                        <p className="text-base text-red-600 mt-2">
                          This action is irreversible and will immediately revoke access for any applications using this
                          key.
                        </p>
                      </div>
                    </div>
                    <p className="text-base text-gray-600 mb-5">Are you sure you want to delete this Virtual Key?</p>
                    <div className="mb-5">
                      <label className="block text-base font-medium text-gray-700 mb-2">
                        {`Type `}
                        <span className="underline">{keyName}</span>
                        {` to confirm deletion:`}
                      </label>
                      <input
                        type="text"
                        value={deleteConfirmInput}
                        onChange={(e) => setDeleteConfirmInput(e.target.value)}
                        placeholder="Enter key name exactly"
                        className="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-base"
                        autoFocus
                      />
                    </div>
                  </div>
                </div>
                <div className="px-6 py-4 bg-gray-50 flex justify-end gap-4">
                  <button
                    onClick={() => {
                      cancelDelete();
                      setDeleteConfirmInput("");
                    }}
                    className="px-5 py-3 bg-white border border-gray-300 rounded-md text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={confirmDelete}
                    disabled={!isValid}
                    className={`px-5 py-3 rounded-md text-base font-medium text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 ${isValid ? "bg-red-600 hover:bg-red-700" : "bg-red-300 cursor-not-allowed"}`}
                  >
                    Delete Virtual Key
                  </button>
                </div>
              </div>
            </div>
          );
        })()}

      {/* Regenerate Key Form Modal */}
      <Modal
        title="Regenerate Virtual Key"
        visible={regenerateDialogVisible}
        onCancel={() => {
          setRegenerateDialogVisible(false);
          regenerateForm.resetFields();
        }}
        footer={[
          <Button
            key="cancel"
            onClick={() => {
              setRegenerateDialogVisible(false);
              regenerateForm.resetFields();
            }}
            className="mr-2"
          >
            Cancel
          </Button>,
          <Button key="regenerate" onClick={handleRegenerateKey} disabled={!premiumUser}>
            {premiumUser ? "Regenerate" : "Upgrade to Regenerate"}
          </Button>,
        ]}
      >
        {premiumUser ? (
          <Form
            form={regenerateForm}
            layout="vertical"
            onValuesChange={(changedValues, allValues) => {
              if ("duration" in changedValues) {
                handleRegenerateFormChange("duration", changedValues.duration);
              }
            }}
          >
            <Form.Item name="key_alias" label="Key Alias">
              <TextInput disabled={true} />
            </Form.Item>
            <Form.Item name="max_budget" label="Max Budget (USD)">
              <InputNumber step={0.01} precision={2} style={{ width: "100%" }} />
            </Form.Item>
            <Form.Item name="tpm_limit" label="TPM Limit">
              <InputNumber style={{ width: "100%" }} />
            </Form.Item>
            <Form.Item name="rpm_limit" label="RPM Limit">
              <InputNumber style={{ width: "100%" }} />
            </Form.Item>
            <Form.Item name="duration" label="Expire Key (eg: 30s, 30h, 30d)" className="mt-8">
              <TextInput placeholder="" />
            </Form.Item>
            <div className="mt-2 text-sm text-gray-500">
              Current expiry:{" "}
              {selectedToken?.expires != null ? new Date(selectedToken.expires).toLocaleString() : "Never"}
            </div>
            {newExpiryTime && <div className="mt-2 text-sm text-green-600">New expiry: {newExpiryTime}</div>}
          </Form>
        ) : (
          <div>
            <p className="mb-2 text-gray-500 italic text-[12px]">Upgrade to use this feature</p>
            <Button variant="primary" className="mb-2">
              <a href="https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat" target="_blank">
                Get Free Trial
              </a>
            </Button>
          </div>
        )}
      </Modal>

      {/* Regenerated Key Display Modal */}
      {regeneratedKey && (
        <Modal
          visible={!!regeneratedKey}
          onCancel={() => setRegeneratedKey(null)}
          footer={[
            <Button key="close" onClick={() => setRegeneratedKey(null)}>
              Close
            </Button>,
          ]}
        >
          <Grid numItems={1} className="gap-2 w-full">
            <Title>Regenerated Key</Title>
            <Col numColSpan={1}>
              <p>
                Please replace your old key with the new key generated. For security reasons,{" "}
                <b>you will not be able to view it again</b> through your LiteLLM account. If you lose this secret key,
                you will need to generate a new one.
              </p>
            </Col>
            <Col numColSpan={1}>
              <Text className="mt-3">Key Alias:</Text>
              <div
                style={{
                  background: "#f8f8f8",
                  padding: "10px",
                  borderRadius: "5px",
                  marginBottom: "10px",
                }}
              >
                <pre style={{ wordWrap: "break-word", whiteSpace: "normal" }}>
                  {selectedToken?.key_alias || "No alias set"}
                </pre>
              </div>
              <Text className="mt-3">New Virtual Key:</Text>
              <div
                style={{
                  background: "#f8f8f8",
                  padding: "10px",
                  borderRadius: "5px",
                  marginBottom: "10px",
                }}
              >
                <pre style={{ wordWrap: "break-word", whiteSpace: "normal" }}>{regeneratedKey}</pre>
              </div>
              <CopyToClipboard
                text={regeneratedKey}
                onCopy={() => NotificationManager.success({ description: "Virtual Key copied to clipboard" })}
              >
                <Button className="mt-3">Copy Virtual Key</Button>
              </CopyToClipboard>
            </Col>
          </Grid>
        </Modal>
      )}
    </div>
  );
};

// Update the type declaration to include the new function
declare global {
  interface Window {
    refreshKeysList?: () => void;
    addNewKeyToList?: (newKey: any) => void;
  }
}

export default ViewKeyTable;
