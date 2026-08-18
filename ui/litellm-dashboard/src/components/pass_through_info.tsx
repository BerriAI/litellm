import React, { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button as AntdButton, Form, Input as AntdInput, Switch, InputNumber, Select } from "antd";
import { updatePassThroughEndpoint, deletePassThroughEndpointsCall } from "./networking";
import { Eye, EyeOff } from "lucide-react";
import RoutePreview from "./route_preview";
import NotificationsManager from "./molecules/notifications_manager";
import PassThroughSecuritySection from "./common_components/PassThroughSecuritySection";
import PassThroughGuardrailsSection from "./common_components/PassThroughGuardrailsSection";

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
        aria-label={showPassword ? "Hide headers" : "Show headers"}
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
          NotificationsManager.fromBackend("Invalid JSON format for headers");
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
      NotificationsManager.fromBackend("Failed to update pass through endpoint");
    }
  };

  const handleDeleteEndpoint = async () => {
    try {
      if (!accessToken || !endpointData?.id) return;

      await deletePassThroughEndpointsCall(accessToken, endpointData.id);
      NotificationsManager.success("Pass through endpoint deleted successfully");
      onClose();
      if (onEndpointUpdated) {
        onEndpointUpdated();
      }
    } catch (error) {
      console.error("Error deleting endpoint:", error);
      NotificationsManager.fromBackend("Failed to delete pass through endpoint");
    }
  };

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  if (!endpointData) {
    return <div className="p-4">Pass through endpoint not found</div>;
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <AntdButton onClick={onClose} className="mb-4">
            ← Back
          </AntdButton>
          <h2 className="text-xl font-semibold">Pass Through Endpoint: {endpointData.path}</h2>
          <p className="text-sm text-gray-500 font-mono">{endpointData.id}</p>
        </div>
      </div>

      <Tabs defaultValue="overview">
        <TabsList variant="line" className="mb-4 h-auto w-full justify-start rounded-none border-b p-0">
          <TabsTrigger value="overview" className="flex-none rounded-none px-4 py-2">
            Overview
          </TabsTrigger>
          {isAdmin && (
            <TabsTrigger value="settings" className="flex-none rounded-none px-4 py-2">
              Settings
            </TabsTrigger>
          )}
        </TabsList>

        <div>
          {/* Overview Panel */}
          <TabsContent value="overview" keepMounted>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card className="block p-6">
                <p className="text-sm">Path</p>
                <div className="mt-2">
                  <h3 className="text-lg font-medium font-mono">{endpointData.path}</h3>
                </div>
              </Card>

              <Card className="block p-6">
                <p className="text-sm">Target</p>
                <div className="mt-2">
                  <h3 className="text-lg font-medium">{endpointData.target}</h3>
                </div>
              </Card>

              <Card className="block p-6">
                <p className="text-sm">Configuration</p>
                <div className="mt-2 space-y-2">
                  <div>
                    <Badge variant={endpointData.include_subpath ? "secondary" : "outline"}>
                      {endpointData.include_subpath ? "Include Subpath" : "Exact Path"}
                    </Badge>
                  </div>
                  <div>
                    <Badge variant={endpointData.auth ? "secondary" : "outline"}>
                      {endpointData.auth ? "Auth Required" : "No Auth"}
                    </Badge>
                  </div>
                  {endpointData.methods && endpointData.methods.length > 0 && (
                    <div>
                      <p className="text-xs text-gray-500">HTTP Methods:</p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {endpointData.methods.map((method) => (
                          <Badge key={method} variant="secondary">
                            {method}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  {(!endpointData.methods || endpointData.methods.length === 0) && (
                    <div>
                      <p className="text-xs text-gray-500">All HTTP methods supported</p>
                    </div>
                  )}
                  {endpointData.cost_per_request !== undefined && (
                    <div>
                      <p className="text-sm">Cost per request: ${endpointData.cost_per_request}</p>
                    </div>
                  )}
                </div>
              </Card>
            </div>

            {/* Route Preview Section */}
            <div className="mt-6">
              <RoutePreview
                pathValue={endpointData.path}
                targetValue={endpointData.target}
                includeSubpath={endpointData.include_subpath || false}
              />
            </div>

            {endpointData.headers && Object.keys(endpointData.headers).length > 0 && (
              <Card className="block mt-6 p-6">
                <div className="flex justify-between items-center">
                  <p className="text-sm font-medium">Headers</p>
                  <Badge variant="secondary">{Object.keys(endpointData.headers).length} headers configured</Badge>
                </div>
                <div className="mt-4">
                  <PasswordField value={endpointData.headers} />
                </div>
              </Card>
            )}

            {endpointData.guardrails && Object.keys(endpointData.guardrails).length > 0 && (
              <Card className="block mt-6 p-6">
                <div className="flex justify-between items-center">
                  <p className="text-sm font-medium">Guardrails</p>
                  <Badge variant="secondary">{Object.keys(endpointData.guardrails).length} guardrails configured</Badge>
                </div>
                <div className="mt-4 space-y-2">
                  {Object.entries(endpointData.guardrails).map(([name, settings]) => (
                    <div key={name} className="p-3 bg-gray-50 rounded-sm">
                      <div className="font-medium text-sm">{name}</div>
                      {settings && (settings.request_fields || settings.response_fields) && (
                        <div className="mt-2 text-xs text-gray-600 space-y-1">
                          {settings.request_fields && <div>Request fields: {settings.request_fields.join(", ")}</div>}
                          {settings.response_fields && (
                            <div>Response fields: {settings.response_fields.join(", ")}</div>
                          )}
                        </div>
                      )}
                      {!settings && <div className="text-xs text-gray-600 mt-1">Uses entire payload</div>}
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </TabsContent>

          {/* Settings Panel (only for admins) */}
          {isAdmin && (
            <TabsContent value="settings" keepMounted>
              <Card className="block p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium">Pass Through Endpoint Settings</h3>
                  <div className="space-x-2">
                    {!isEditing && (
                      <>
                        <Button onClick={() => setIsEditing(true)}>Edit Settings</Button>
                        <Button onClick={handleDeleteEndpoint} variant="destructive">
                          Delete Endpoint
                        </Button>
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
                      label="Target URL"
                      name="target"
                      rules={[{ required: true, message: "Please input a target URL" }]}
                    >
                      <Input placeholder="https://api.example.com" />
                    </Form.Item>

                    <Form.Item label="Headers (JSON)" name="headers">
                      <AntdInput.TextArea
                        rows={5}
                        placeholder='{"Authorization": "Bearer your-token", "Content-Type": "application/json"}'
                      />
                    </Form.Item>

                    <Form.Item
                      label="HTTP Methods (Optional)"
                      name="methods"
                      extra={
                        selectedMethods.length === 0
                          ? "All HTTP methods supported (default)"
                          : `Only ${selectedMethods.join(", ")} requests will be routed to this endpoint`
                      }
                    >
                      <Select
                        mode="multiple"
                        placeholder="Select methods (leave empty for all)"
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

                    <Form.Item label="Include Subpath" name="include_subpath" valuePropName="checked">
                      <Switch />
                    </Form.Item>

                    <Form.Item label="Cost per Request" name="cost_per_request">
                      <InputNumber min={0} step={0.01} precision={2} placeholder="0.00" addonBefore="$" />
                    </Form.Item>

                    <Form.Item
                      label="Request Timeout (seconds)"
                      name="timeout"
                      extra="Max time to wait for upstream response. Leave empty to use the global pass_through_request_timeout (default 600s)."
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
                      <AntdButton onClick={() => setIsEditing(false)}>Cancel</AntdButton>
                      <Button type="submit">Save Changes</Button>
                    </div>
                  </Form>
                ) : (
                  <div className="space-y-4">
                    <div>
                      <p className="text-sm font-medium">Path</p>
                      <div className="font-mono">{endpointData.path}</div>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Target URL</p>
                      <div>{endpointData.target}</div>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Include Subpath</p>
                      <Badge variant={endpointData.include_subpath ? "secondary" : "outline"}>
                        {endpointData.include_subpath ? "Yes" : "No"}
                      </Badge>
                    </div>
                    {endpointData.cost_per_request !== undefined && (
                      <div>
                        <p className="text-sm font-medium">Cost per Request</p>
                        <div>${endpointData.cost_per_request}</div>
                      </div>
                    )}
                    {endpointData.timeout !== undefined && endpointData.timeout !== null && (
                      <div>
                        <p className="text-sm font-medium">Request Timeout</p>
                        <div>{endpointData.timeout}s</div>
                      </div>
                    )}
                    <div>
                      <p className="text-sm font-medium">Authentication Required</p>
                      <Badge variant={endpointData.auth ? "secondary" : "outline"}>
                        {endpointData.auth ? "Yes" : "No"}
                      </Badge>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Headers</p>
                      {endpointData.headers && Object.keys(endpointData.headers).length > 0 ? (
                        <div className="mt-2">
                          <PasswordField value={endpointData.headers} />
                        </div>
                      ) : (
                        <div className="text-gray-500">No headers configured</div>
                      )}
                    </div>
                  </div>
                )}
              </Card>
            </TabsContent>
          )}
        </div>
      </Tabs>
    </div>
  );
};

export default PassThroughInfoView;
