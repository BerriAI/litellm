"use client";

import { Button as Button2, Form, FormInstance, Modal } from "antd";
import { Text } from "@tremor/react";
import { keyCreateCall, keyCreateServiceAccountCall } from "@/components/networking";
import React, { useEffect, useState } from "react";
import { Team } from "@/components/key_team_helpers/key_list";
import SaveKeyModal from "@/app/(dashboard)/virtual-keys/components/SaveKeyModal";
import NotificationsManager from "@/components/molecules/notifications_manager";
import {
  useGuardrailsAndPrompts,
  useMcpAccessGroups,
  useTeamModels,
  useUserModels,
  useUserSearch,
} from "@/app/(dashboard)/virtual-keys/components/CreateKeyModal/hooks";
import OwnershipSection from "@/app/(dashboard)/virtual-keys/components/CreateKeyModal/CreateKeyForm/OwnershipSection";
import KeyDetailsSection from "@/app/(dashboard)/virtual-keys/components/CreateKeyModal/CreateKeyForm/KeyDetailsSection";
import OptionalSettingsSection from "@/app/(dashboard)/virtual-keys/components/CreateKeyModal/CreateKeyForm/OptionalSettingsSection";
import { ModelAliases } from "@/app/(dashboard)/virtual-keys/components/CreateKeyModal/types";
import { getPredefinedTags, prepareFormValues } from "@/app/(dashboard)/virtual-keys/components/CreateKeyModal/utils";
import { fetchTeams } from "@/app/(dashboard)/virtual-keys/networking";
import useTeams from "@/app/(dashboard)/virtual-keys/hooks/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export interface CreateKeyModalProps {
  isModalVisible: boolean;
  setIsModalVisible: (isModalVisible: boolean) => void;
  form: FormInstance;
  handleUserSelect: (_value: string, option: UserOption) => void;
  setIsCreateUserModalVisible: (visible: boolean) => void;
  team: Team | null;
  data: any[] | null;
  addKey: (data: any) => void;
}

interface User {
  user_id: string;
  user_email: string;
  role?: string;
}

interface UserOption {
  label: string;
  value: string;
  user: User;
}

const CreateKeyModal = ({
  isModalVisible,
  setIsModalVisible,
  form,
  handleUserSelect,
  setIsCreateUserModalVisible,
  team,
  data,
  addKey,
}: CreateKeyModalProps) => {
  const { userId: userID, userRole, accessToken, premiumUser } = useAuthorized();
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [keyOwner, setKeyOwner] = useState("you");
  const [keyType, setKeyType] = useState<string>("default");

  const [selectedCreateKeyTeam, setSelectedCreateKeyTeam] = useState<Team | null>(team);

  const [softBudget, setSoftBudget] = useState<string | null>(null);
  const [loggingSettings, setLoggingSettings] = useState<any[]>([]);
  const [disabledCallbacks, setDisabledCallbacks] = useState<string[]>([]);
  const [autoRotationEnabled, setAutoRotationEnabled] = useState<boolean>(false);
  const [rotationInterval, setRotationInterval] = useState<string>("30d");
  const [modelAliases, setModelAliases] = useState<ModelAliases>({});
  const predefinedTags = getPredefinedTags(data);

  const { options: userOptions, loading: userSearchLoading, onSearch: handleUserSearch } = useUserSearch();
  const { guardrails, prompts } = useGuardrailsAndPrompts();
  const teams = useTeams();
  const mcpAccessGroups = useMcpAccessGroups();
  const userModels = useUserModels();
  const modelsToPick = useTeamModels(selectedCreateKeyTeam);

  const isTeamSelectionRequired = modelsToPick.includes("no-default-models");
  const isFormDisabled = isTeamSelectionRequired && !selectedCreateKeyTeam;

  const handleCancel = () => {
    setIsModalVisible(false);
    setApiKey(null);
    setSelectedCreateKeyTeam(null);
    form.resetFields();
    setLoggingSettings([]);
    setDisabledCallbacks([]);
    setKeyType("default");
    setModelAliases({});
    setAutoRotationEnabled(false);
    setRotationInterval("30d");
  };

  const handleOk = () => {
    setIsModalVisible(false);
    form.resetFields();
    setLoggingSettings([]);
    setDisabledCallbacks([]);
    setKeyType("default");
    setModelAliases({});
    setAutoRotationEnabled(false);
    setRotationInterval("30d");
  };

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      const newKeyAlias = formValues?.key_alias ?? "";
      const newKeyTeamId = formValues?.team_id ?? null;

      const existingKeyAliases = data?.filter((k) => k.team_id === newKeyTeamId).map((k) => k.key_alias) ?? [];

      if (existingKeyAliases.includes(newKeyAlias)) {
        throw new Error(
          `Key alias ${newKeyAlias} already exists for team with ID ${newKeyTeamId}, please provide another key alias`,
        );
      }

      NotificationsManager.info("Making API Call");
      setIsModalVisible(true);

      const prepared = prepareFormValues(formValues, {
        keyOwner,
        userID,
        loggingSettings,
        disabledCallbacks,
        autoRotationEnabled,
        rotationInterval,
        modelAliases,
      });

      let response;
      if (keyOwner === "service_account") {
        response = await keyCreateServiceAccountCall(accessToken, prepared);
      } else {
        response = await keyCreateCall(accessToken, userID, prepared);
      }

      // TODO: change logic to trigger API call in parent component
      addKey(response);

      setApiKey(response["key"]);
      setSoftBudget(response["soft_budget"]);
      NotificationsManager.success("API Key Created");
      form.resetFields();
      localStorage.removeItem("userData" + userID);
    } catch (error) {
      console.log("error in create key:", error);
      NotificationsManager.fromBackend(`Error creating the key: ${error}`);
    }
  };

  useEffect(() => {
    form.setFieldValue("models", []);
  }, [selectedCreateKeyTeam, accessToken, userID, userRole, form]);

  return (
    <div>
      <Modal open={isModalVisible} width={1000} footer={null} onOk={handleOk} onCancel={handleCancel}>
        <Form form={form} onFinish={handleCreate} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} labelAlign="left">
          <OwnershipSection
            team={team}
            teams={teams}
            userRole={userRole}
            userOptions={userOptions}
            keyOwner={keyOwner}
            setKeyOwner={setKeyOwner}
            handleUserSearch={handleUserSearch}
            userSearchLoading={userSearchLoading}
            handleUserSelect={handleUserSelect}
            setSelectedCreateKeyTeam={setSelectedCreateKeyTeam}
            setIsCreateUserModalVisible={setIsCreateUserModalVisible}
          />

          {/* Show message when team selection is required */}
          {isFormDisabled && (
            <div className="mb-8 p-4 bg-blue-50 border border-blue-200 rounded-md">
              <Text className="text-blue-800 text-sm">
                Please select a team to continue configuring your API key. If you do not see any teams, please contact
                your Proxy Admin to either provide you with access to models or to add you to a team.
              </Text>
            </div>
          )}

          {/* Section 2: Key Details */}
          {!isFormDisabled && (
            <KeyDetailsSection
              form={form}
              keyOwner={keyOwner}
              modelsToPick={modelsToPick}
              keyType={keyType}
              setKeyType={setKeyType}
            />
          )}

          {/* Section 3: Optional Settings */}
          {!isFormDisabled && (
            <OptionalSettingsSection
              form={form}
              team={team}
              premiumUser={premiumUser}
              guardrails={guardrails}
              prompts={prompts}
              accessToken={accessToken}
              predefinedTags={predefinedTags}
              loggingSettings={loggingSettings}
              setLoggingSettings={setLoggingSettings}
              disabledCallbacks={disabledCallbacks}
              setDisabledCallbacks={setDisabledCallbacks}
              modelAliases={modelAliases}
              setModelAliases={setModelAliases}
              autoRotationEnabled={autoRotationEnabled}
              setAutoRotationEnabled={setAutoRotationEnabled}
              rotationInterval={rotationInterval}
              setRotationInterval={setRotationInterval}
            />
          )}

          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit" disabled={isFormDisabled} style={{ opacity: isFormDisabled ? 0.5 : 1 }}>
              Create Key
            </Button2>
          </div>
        </Form>
      </Modal>
      {apiKey && (
        <SaveKeyModal apiKey={apiKey} isModalVisible={isModalVisible} handleOk={handleOk} handleCancel={handleCancel} />
      )}
    </div>
  );
};

export default CreateKeyModal;
