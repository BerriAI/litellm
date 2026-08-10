import React, { useState } from "react";
import {
  Card,
  Title,
  Text,
  Grid,
  Badge,
  Button as TremorButton,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
  TextInput,
} from "@tremor/react";
import { Button, Form, Input, Switch, InputNumber, Select } from "antd";
import { updatePassThroughEndpoint, deletePassThroughEndpointsCall } from "./networking";
import { Eye, EyeOff } from "lucide-react";
import RoutePreview from "./route_preview";
import NotificationsManager from "./molecules/notifications_manager";
import PassThroughSecuritySection from "./common_components/PassThroughSecuritySection";
import PassThroughGuardrailsSection from "./common_components/PassThroughGuardrailsSection";
import { useTranslation } from "react-i18next";

const HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"];
const { Option } = Select;

export interface PassThroughInfoProps {
  endpointData: PassThroughEndpoint;
  onClose: () => void;
  accessToken: string | null;
  isAdmin: boolean;
  premiumUser?: boolean;
  onEndpointUpdated?: () => void;
}

interface PassThroughEndpoint {
  id?: string;
  path: string;
  target: string;
  headers: Record<string, any>;
  include_subpath?: boolean;
  cost_per_request?: number;
  timeout?: number;
  auth?: boolean;
  methods?: string[];
  guardrails?: Record<string, { request_fields?: string[]; response_fields?: string[] } | null>;
}

// Password field component for headers
const PasswordField: React.FC<{ value: Record<string, any> }> = ({ value }) => {
  const { t } = useTranslation("gateway");
  const [showPassword, setShowPassword] = useState(false);
  const headerString = JSON.stringify(value, null, 2);

  return (
    <div className="flex items-center space-x-2">
      <pre className="font-mono text-xs bg-gray-50 p-2 rounded-sm max-w-md overflow-auto">
        {showPassword ? headerString : "••••••••"}
      </pre>
      <button
        onClick={() => setShowPassword(!showPassword)}
        className="p-1 hover:bg-gray-100 rounded-sm"
        type="button"
        aria-label={t(showPassword ? "models.passThrough.info.hideHeaders" : "models.passThrough.info.showHeaders")}
      >
        {showPassword ? <EyeOff className="w-4 h-4 text-gray-500" /> : <Eye className="w-4 h-4 text-gray-500" />}
      </button>
    </div>
  );
};

const PassThroughInfoView: React.FC<PassThroughInfoProps> = ({
  endpointData: initialEndpointData,
  onClose,
  accessToken,
  isAdmin,
  premiumUser = false,
  onEndpointUpdated,
}) => {
  const { t } = useTranslation("gateway");
  const [endpointData, setEndpointData] = useState<PassThroughEndpoint | null>(initialEndpointData);
  const [loading] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [authEnabled, setAuthEnabled] = useState(initialEndpointData?.auth || false);
  const [selectedMethods, setSelectedMethods] = useState<string[]>(initialEndpointData?.methods || []);
  const [guardrails, setGuardrails] = useState<
    Record<string, { request_fields?: string[]; response_fields?: string[] } | null>
  >(initialEndpointData?.guardrails || {});
  const [form] = Form.useForm();

  const handleEndpointUpdate = async (values: any) => {
    try {
      if (!accessToken || !endpointData?.id) return;

      // Parse headers if provided as string
      let headers = {};
      if (values.headers) {
        try {
          headers = typeof values.headers === "string" ? JSON.parse(values.headers) : values.headers;
        } catch (e) {
          NotificationsManager.fromBackend(t("models.passThrough.info.invalidHeaders"));
          return;
        }
      }

      const updateData = {
        path: endpointData.path,
        target: values.target,
        headers: headers,
        include_subpath: values.include_subpath,
        cost_per_request: values.cost_per_request,
        timeout: values.timeout,
        auth: premiumUser ? values.auth : undefined,
        methods: selectedMethods && selectedMethods.length > 0 ? selectedMethods : undefined,
        guardrails: guardrails && Object.keys(guardrails).length > 0 ? guardrails : undefined,
      };

      await updatePassThroughEndpoint(accessToken, endpointData.id, updateData);

      // Update local state with the new values
      setEndpointData({
        ...endpointData,
        ...updateData,
      });

      setIsEditing(false);
      if (onEndpointUpdated) {
        onEndpointUpdated();
      }
    } catch (error) {
      console.error("Error updating endpoint:", error);
      NotificationsManager.fromBackend(t("models.passThrough.info.updateFailed"));
    }
  };

  const handleDeleteEndpoint = async () => {
    try {
      if (!accessToken || !endpointData?.id) return;

      await deletePassThroughEndpointsCall(accessToken, endpointData.id);
      NotificationsManager.success(t("models.passThrough.info.deleted"));
      onClose();
      if (onEndpointUpdated) {
        onEndpointUpdated();
      }
    } catch (error) {
      console.error("Error deleting endpoint:", error);
      NotificationsManager.fromBackend(t("models.passThrough.info.deleteFailed"));
    }
  };

  if (loading) {
    return <div className="p-4">{t("models.passThrough.info.loading")}</div>;
  }

  if (!endpointData) {
    return <div className="p-4">{t("models.passThrough.info.notFound")}</div>;
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button onClick={onClose} className="mb-4">
            {t("models.passThrough.info.back")}
          </Button>
          <Title>{t("models.passThrough.info.title", { path: endpointData.path })}</Title>
          <Text className="text-gray-500 font-mono">{endpointData.id}</Text>
        </div>
      </div>

      <TabGroup>
        <TabList className="mb-4">
          <Tab key="overview">{t("models.passThrough.info.overview")}</Tab>
          {isAdmin ? <Tab key="settings">{t("models.passThrough.info.settings")}</Tab> : <></>}
        </TabList>

        <TabPanels>
          {/* Overview Panel */}
          <TabPanel>
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
              <Card>
                <Text>{t("models.passThrough.info.path")}</Text>
                <div className="mt-2">
                  <Title className="font-mono">{endpointData.path}</Title>
                </div>
              </Card>

              <Card>
                <Text>{t("models.passThrough.info.target")}</Text>
                <div className="mt-2">
                  <Title>{endpointData.target}</Title>
                </div>
              </Card>

              <Card>
                <Text>{t("models.passThrough.info.configuration")}</Text>
                <div className="mt-2 space-y-2">
                  <div>
                    <Badge color={endpointData.include_subpath ? "green" : "gray"}>
                      {t(
                        endpointData.include_subpath
                          ? "models.passThrough.info.includeSubpath"
                          : "models.passThrough.info.exactPath",
                      )}
                    </Badge>
                  </div>
                  <div>
                    <Badge color={endpointData.auth ? "blue" : "gray"}>
                      {t(endpointData.auth ? "models.passThrough.info.authRequired" : "models.passThrough.info.noAuth")}
                    </Badge>
                  </div>
                  {endpointData.methods && endpointData.methods.length > 0 && (
                    <div>
                      <Text className="text-xs text-gray-500">{t("models.passThrough.info.methods")}</Text>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {endpointData.methods.map((method) => (
                          <Badge key={method} color="indigo" size="sm">
                            {method}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  {(!endpointData.methods || endpointData.methods.length === 0) && (
                    <div>
                      <Text className="text-xs text-gray-500">{t("models.passThrough.info.allMethods")}</Text>
                    </div>
                  )}
                  {endpointData.cost_per_request !== undefined && (
                    <div>
                      <Text>
                        {t("models.passThrough.info.costPerRequest", { cost: endpointData.cost_per_request })}
                      </Text>
                    </div>
                  )}
                </div>
              </Card>
            </Grid>

            {/* Route Preview Section */}
            <div className="mt-6">
              <RoutePreview
                pathValue={endpointData.path}
                targetValue={endpointData.target}
                includeSubpath={endpointData.include_subpath || false}
              />
            </div>

            {endpointData.headers && Object.keys(endpointData.headers).length > 0 && (
              <Card className="mt-6">
                <div className="flex justify-between items-center">
                  <Text className="font-medium">{t("models.passThrough.info.headers")}</Text>
                  <Badge color="blue">
                    {t("models.passThrough.info.headersConfigured", {
                      count: Object.keys(endpointData.headers).length,
                    })}
                  </Badge>
                </div>
                <div className="mt-4">
                  <PasswordField value={endpointData.headers} />
                </div>
              </Card>
            )}

            {endpointData.guardrails && Object.keys(endpointData.guardrails).length > 0 && (
              <Card className="mt-6">
                <div className="flex justify-between items-center">
                  <Text className="font-medium">{t("models.passThrough.info.guardrails")}</Text>
                  <Badge color="purple">
                    {t("models.passThrough.info.guardrailsConfigured", {
                      count: Object.keys(endpointData.guardrails).length,
                    })}
                  </Badge>
                </div>
                <div className="mt-4 space-y-2">
                  {Object.entries(endpointData.guardrails).map(([name, settings]) => (
                    <div key={name} className="p-3 bg-gray-50 rounded-sm">
                      <div className="font-medium text-sm">{name}</div>
                      {settings && (settings.request_fields || settings.response_fields) && (
                        <div className="mt-2 text-xs text-gray-600 space-y-1">
                          {settings.request_fields && (
                            <div>
                              {t("models.passThrough.info.requestFields", {
                                fields: settings.request_fields.join(", "),
                              })}
                            </div>
                          )}
                          {settings.response_fields && (
                            <div>
                              {t("models.passThrough.info.responseFields", {
                                fields: settings.response_fields.join(", "),
                              })}
                            </div>
                          )}
                        </div>
                      )}
                      {!settings && (
                        <div className="text-xs text-gray-600 mt-1">{t("models.passThrough.info.entirePayload")}</div>
                      )}
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </TabPanel>

          {/* Settings Panel (only for admins) */}
          {isAdmin && (
            <TabPanel>
              <Card>
                <div className="flex justify-between items-center mb-4">
                  <Title>{t("models.passThrough.info.settingsTitle")}</Title>
                  <div className="space-x-2">
                    {!isEditing && (
                      <>
                        <TremorButton onClick={() => setIsEditing(true)}>
                          {t("models.passThrough.info.editSettings")}
                        </TremorButton>
                        <TremorButton onClick={handleDeleteEndpoint} variant="secondary" color="red">
                          {t("models.passThrough.info.deleteEndpoint")}
                        </TremorButton>
                      </>
                    )}
                  </div>
                </div>

                {isEditing ? (
                  <Form
                    form={form}
                    onFinish={handleEndpointUpdate}
                    initialValues={{
                      target: endpointData.target,
                      headers: endpointData.headers ? JSON.stringify(endpointData.headers, null, 2) : "",
                      include_subpath: endpointData.include_subpath || false,
                      cost_per_request: endpointData.cost_per_request,
                      timeout: endpointData.timeout,
                      auth: endpointData.auth || false,
                      methods: endpointData.methods || [],
                    }}
                    layout="vertical"
                  >
                    <Form.Item
                      label={t("models.passThrough.info.targetUrl")}
                      name="target"
                      rules={[{ required: true, message: t("models.passThrough.info.targetRequired") }]}
                    >
                      <TextInput placeholder="https://api.example.com" />
                    </Form.Item>

                    <Form.Item label={t("models.passThrough.info.headersJson")} name="headers">
                      <Input.TextArea
                        rows={5}
                        placeholder='{"Authorization": "Bearer your-token", "Content-Type": "application/json"}'
                      />
                    </Form.Item>

                    <Form.Item
                      label={t("models.passThrough.form.methods")}
                      name="methods"
                      extra={
                        selectedMethods.length === 0
                          ? t("models.passThrough.form.allMethods")
                          : t("models.passThrough.form.selectedMethods", { methods: selectedMethods.join(", ") })
                      }
                    >
                      <Select
                        mode="multiple"
                        placeholder={t("models.passThrough.form.methodsPlaceholder")}
                        value={selectedMethods}
                        onChange={setSelectedMethods}
                        allowClear
                        style={{ width: "100%" }}
                      >
                        {HTTP_METHODS.map((method) => (
                          <Option key={method} value={method}>
                            {method}
                          </Option>
                        ))}
                      </Select>
                    </Form.Item>

                    <Form.Item
                      label={t("models.passThrough.info.includeSubpath")}
                      name="include_subpath"
                      valuePropName="checked"
                    >
                      <Switch />
                    </Form.Item>

                    <Form.Item label={t("models.passThrough.info.cost")} name="cost_per_request">
                      <InputNumber min={0} step={0.01} precision={2} placeholder="0.00" addonBefore="$" />
                    </Form.Item>

                    <Form.Item
                      label={t("models.passThrough.form.timeout")}
                      name="timeout"
                      extra={t("models.passThrough.form.timeoutTooltip")}
                    >
                      <InputNumber min={1} step={1} precision={0} placeholder="600" style={{ width: "100%" }} />
                    </Form.Item>

                    <PassThroughSecuritySection
                      premiumUser={premiumUser}
                      authEnabled={authEnabled}
                      onAuthChange={(checked) => {
                        setAuthEnabled(checked);
                        form.setFieldsValue({ auth: checked });
                      }}
                    />

                    <div className="mt-4">
                      <PassThroughGuardrailsSection
                        accessToken={accessToken || ""}
                        value={guardrails}
                        onChange={setGuardrails}
                      />
                    </div>

                    <div className="flex justify-end gap-2 mt-6">
                      <Button onClick={() => setIsEditing(false)}>{t("models.passThrough.info.cancel")}</Button>
                      <TremorButton>{t("models.passThrough.info.save")}</TremorButton>
                    </div>
                  </Form>
                ) : (
                  <div className="space-y-4">
                    <div>
                      <Text className="font-medium">{t("models.passThrough.info.path")}</Text>
                      <div className="font-mono">{endpointData.path}</div>
                    </div>
                    <div>
                      <Text className="font-medium">{t("models.passThrough.info.targetUrl")}</Text>
                      <div>{endpointData.target}</div>
                    </div>
                    <div>
                      <Text className="font-medium">{t("models.passThrough.info.includeSubpath")}</Text>
                      <Badge color={endpointData.include_subpath ? "green" : "gray"}>
                        {t(
                          endpointData.include_subpath
                            ? "models.passThrough.values.yes"
                            : "models.passThrough.values.no",
                        )}
                      </Badge>
                    </div>
                    {endpointData.cost_per_request !== undefined && (
                      <div>
                        <Text className="font-medium">{t("models.passThrough.info.cost")}</Text>
                        <div>${endpointData.cost_per_request}</div>
                      </div>
                    )}
                    {endpointData.timeout !== undefined && endpointData.timeout !== null && (
                      <div>
                        <Text className="font-medium">{t("models.passThrough.info.timeout")}</Text>
                        <div>{endpointData.timeout}s</div>
                      </div>
                    )}
                    <div>
                      <Text className="font-medium">{t("models.passThrough.info.authenticationRequired")}</Text>
                      <Badge color={endpointData.auth ? "green" : "gray"}>
                        {t(endpointData.auth ? "models.passThrough.values.yes" : "models.passThrough.values.no")}
                      </Badge>
                    </div>
                    <div>
                      <Text className="font-medium">{t("models.passThrough.info.headers")}</Text>
                      {endpointData.headers && Object.keys(endpointData.headers).length > 0 ? (
                        <div className="mt-2">
                          <PasswordField value={endpointData.headers} />
                        </div>
                      ) : (
                        <div className="text-gray-500">{t("models.passThrough.info.noHeaders")}</div>
                      )}
                    </div>
                  </div>
                )}
              </Card>
            </TabPanel>
          )}
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default PassThroughInfoView;
