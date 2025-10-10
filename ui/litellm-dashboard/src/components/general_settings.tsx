import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Table,
  TableHead,
  TableRow,
  Badge,
  TableHeaderCell,
  TableCell,
  TableBody,
  Text,
  Grid,
  Button,
  TextInput,
  Select as Select2,
  SelectItem,
  Col,
  Accordion,
  AccordionBody,
  AccordionHeader,
} from "@tremor/react";
import { TabPanel, TabPanels, TabGroup, TabList, Tab, Icon } from "@tremor/react";
import {
  getCallbacksCall,
  setCallbacksCall,
  getGeneralSettingsCall,
  updateConfigFieldSetting,
  deleteConfigFieldSetting,
} from "./networking";
import { Form, InputNumber } from "antd";
import { TrashIcon, CheckCircleIcon } from "@heroicons/react/outline";

import AddFallbacks from "./add_fallbacks";
import openai from "openai";
import NotificationsManager from "./molecules/notifications_manager";
interface GeneralSettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  modelData: any;
}

async function testFallbackModelResponse(selectedModel: string, accessToken: string) {
  // base url should be the current base_url
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal != true) {
    console.log = function () {};
  }
  console.log("isLocal:", isLocal);
  const proxyBaseUrl = isLocal ? "http://localhost:4000" : window.location.origin;
  const client = new openai.OpenAI({
    apiKey: accessToken, // Replace with your OpenAI API key
    baseURL: proxyBaseUrl, // Replace with your OpenAI API base URL
    dangerouslyAllowBrowser: true, // using a temporary litellm proxy key
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

interface AccordionHeroProps {
  selectedStrategy: string | null;
  strategyArgs: routingStrategyArgs;
  paramExplanation: { [key: string]: string };
}

interface routingStrategyArgs {
  ttl?: number;
  lowest_latency_buffer?: number;
}

interface generalSettingsItem {
  field_name: string;
  field_type: string;
  field_value: any;
  field_description: string;
  stored_in_db: boolean | null;
}

const defaultLowestLatencyArgs: routingStrategyArgs = {
  ttl: 3600,
  lowest_latency_buffer: 0,
};

export const AccordionHero: React.FC<AccordionHeroProps> = ({ selectedStrategy, strategyArgs, paramExplanation }) => (
  <Accordion>
    <AccordionHeader className="text-sm font-medium text-tremor-content-strong dark:text-dark-tremor-content-strong">
      Routing Strategy Specific Args
    </AccordionHeader>
    <AccordionBody>
      {selectedStrategy == "latency-based-routing" ? (
        <Card>
          <Table>
            <TableHead>
              <TableRow>
                <TableHeaderCell>Setting</TableHeaderCell>
                <TableHeaderCell>Value</TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(strategyArgs).map(([param, value]) => (
                <TableRow key={param}>
                  <TableCell>
                    <Text>{param}</Text>
                    <p
                      style={{
                        fontSize: "0.65rem",
                        color: "#808080",
                        fontStyle: "italic",
                      }}
                      className="mt-1"
                    >
                      {paramExplanation[param]}
                    </p>
                  </TableCell>
                  <TableCell>
                    <TextInput
                      name={param}
                      defaultValue={typeof value === "object" ? JSON.stringify(value, null, 2) : value.toString()}
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      ) : (
        <Text>No specific settings</Text>
      )}
    </AccordionBody>
  </Accordion>
);

const GeneralSettings: React.FC<GeneralSettingsPageProps> = ({ accessToken, userRole, userID, modelData }) => {
  const [routerSettings, setRouterSettings] = useState<{ [key: string]: any }>({});
  const [generalSettingsDict, setGeneralSettingsDict] = useState<{
    [key: string]: any;
  }>({});
  const [generalSettings, setGeneralSettings] = useState<generalSettingsItem[]>([]);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [selectedCallback, setSelectedCallback] = useState<string | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);
  const [strategySettings, setStrategySettings] = useState<routingStrategyArgs | null>(null);

  let paramExplanation: { [key: string]: string } = {
    routing_strategy_args: "(dict) Arguments to pass to the routing strategy",
    routing_strategy: "(string) Routing strategy to use",
    allowed_fails: "(int) Number of times a deployment can fail before being added to cooldown",
    cooldown_time: "(int) time in seconds to cooldown a deployment after failure",
    num_retries: "(int) Number of retries for failed requests. Defaults to 0.",
    timeout: "(float) Timeout for requests. Defaults to None.",
    retry_after: "(int) Minimum time to wait before retrying a failed request",
    ttl: "(int) Sliding window to look back over when calculating the average latency of a deployment. Default - 1 hour (in seconds).",
    lowest_latency_buffer:
      "(float) Shuffle between deployments within this % of the lowest latency. Default - 0 (i.e. always pick lowest latency).",
  };

  useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
    getCallbacksCall(accessToken, userID, userRole).then((data) => {
      console.log("callbacks", data);
      let router_settings = data.router_settings;
      // remove "model_group_retry_policy" from general_settings if exists
      if ("model_group_retry_policy" in router_settings) {
        delete router_settings["model_group_retry_policy"];
      }
      setRouterSettings(router_settings);
    });
    getGeneralSettingsCall(accessToken).then((data) => {
      let general_settings = data;
      setGeneralSettings(general_settings);
    });
  }, [accessToken, userRole, userID]);

  const handleAddCallback = () => {
    console.log("Add callback clicked");
    setIsModalVisible(true);
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    form.resetFields();
    setSelectedCallback(null);
  };

  const deleteFallbacks = async (key: string) => {
    /**
     * pop the key from the Object, if it exists
     */
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

  const handleInputChange = (fieldName: string, newValue: any) => {
    // Update the value in the state
    const updatedSettings = generalSettings.map((setting) =>
      setting.field_name === fieldName ? { ...setting, field_value: newValue } : setting,
    );
    setGeneralSettings(updatedSettings);
  };

  const handleUpdateField = (fieldName: string, idx: number) => {
    if (!accessToken) {
      return;
    }

    let fieldValue = generalSettings[idx].field_value;

    if (fieldValue == null || fieldValue == undefined) {
      return;
    }
    try {
      updateConfigFieldSetting(accessToken, fieldName, fieldValue);
      // update value in state

      const updatedSettings = generalSettings.map((setting) =>
        setting.field_name === fieldName ? { ...setting, stored_in_db: true } : setting,
      );
      setGeneralSettings(updatedSettings);
    } catch (error) {
      // do something
    }
  };

  const handleResetField = (fieldName: string, idx: number) => {
    if (!accessToken) {
      return;
    }

    try {
      deleteConfigFieldSetting(accessToken, fieldName);
      // update value in state

      const updatedSettings = generalSettings.map((setting) =>
        setting.field_name === fieldName ? { ...setting, stored_in_db: null, field_value: null } : setting,
      );
      setGeneralSettings(updatedSettings);
    } catch (error) {
      // do something
    }
  };

  const handleSaveChanges = (router_settings: any) => {
    if (!accessToken) {
      return;
    }

    console.log("router_settings", router_settings);

    const numberKeys = new Set(["allowed_fails", "cooldown_time", "num_retries", "timeout", "retry_after"]);
    const jsonKeys = new Set(["model_group_alias", "retry_policy"]);

    const parseInputValue = (key: string, raw: string | undefined, fallback: unknown) => {
      if (raw === undefined) return fallback;

      const v = raw.trim();

      if (v.toLowerCase() === "null") return null;

      if (numberKeys.has(key)) {
        const n = Number(v);
        return Number.isNaN(n) ? fallback : n;
      }

      if (jsonKeys.has(key)) {
        if (v === "") return null;
        try {
          return JSON.parse(v);
        } catch {
          return fallback;
        }
      }

      if (v.toLowerCase() === "true") return true;
      if (v.toLowerCase() === "false") return false;

      return v;
    };

    const updatedVariables = Object.fromEntries(
      Object.entries(router_settings)
        .map(([key, value]) => {
          if (key !== "routing_strategy_args" && key !== "routing_strategy") {
            const inputEl = document.querySelector(`input[name="${key}"]`) as HTMLInputElement | null;
            const parsed = parseInputValue(key, inputEl?.value, value);
            return [key, parsed];
          } else if (key === "routing_strategy") {
            return [key, selectedStrategy];
          } else if (key === "routing_strategy_args" && selectedStrategy === "latency-based-routing") {
            let setRoutingStrategyArgs: routingStrategyArgs = {};

            const lowestLatencyBufferElement = document.querySelector(
              `input[name="lowest_latency_buffer"]`,
            ) as HTMLInputElement;
            const ttlElement = document.querySelector(`input[name="ttl"]`) as HTMLInputElement;

            if (lowestLatencyBufferElement?.value) {
              setRoutingStrategyArgs["lowest_latency_buffer"] = Number(lowestLatencyBufferElement.value);
            }

            if (ttlElement?.value) {
              setRoutingStrategyArgs["ttl"] = Number(ttlElement.value);
            }

            console.log(`setRoutingStrategyArgs: ${setRoutingStrategyArgs}`);
            return ["routing_strategy_args", setRoutingStrategyArgs];
          }
          return null;
        })
        .filter((entry) => entry !== null && entry !== undefined) as Iterable<[string, unknown]>,
    );
    console.log("updatedVariables", updatedVariables);

    const payload = {
      router_settings: updatedVariables,
    };

    try {
      setCallbacksCall(accessToken, payload);
    } catch (error) {
      NotificationsManager.fromBackend("Failed to update router settings: " + error);
    }

    NotificationsManager.success("router settings updated successfully");
  };

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full mx-4">
      <TabGroup className="gap-2 p-8 h-[75vh] w-full mt-2">
        <TabList variant="line" defaultValue="1">
          <Tab value="1">Loadbalancing</Tab>
          <Tab value="2">Fallbacks</Tab>
          <Tab value="3">General</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <Grid numItems={1} className="gap-2 p-8 w-full mt-2">
              <Title>Router Settings</Title>
              <Card>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>Setting</TableHeaderCell>
                      <TableHeaderCell>Value</TableHeaderCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(routerSettings)
                      .filter(
                        ([param, value]) =>
                          param != "fallbacks" &&
                          param != "context_window_fallbacks" &&
                          param != "routing_strategy_args",
                      )
                      .map(([param, value]) => (
                        <TableRow key={param}>
                          <TableCell>
                            <Text>{param}</Text>
                            <p
                              style={{
                                fontSize: "0.65rem",
                                color: "#808080",
                                fontStyle: "italic",
                              }}
                              className="mt-1"
                            >
                              {paramExplanation[param]}
                            </p>
                          </TableCell>
                          <TableCell>
                            {param == "routing_strategy" ? (
                              <Select2
                                defaultValue={value}
                                className="w-full max-w-md"
                                onValueChange={setSelectedStrategy}
                              >
                                <SelectItem value="usage-based-routing">usage-based-routing</SelectItem>
                                <SelectItem value="latency-based-routing">latency-based-routing</SelectItem>
                                <SelectItem value="simple-shuffle">simple-shuffle</SelectItem>
                              </Select2>
                            ) : (
                              <TextInput
                                name={param}
                                defaultValue={
                                  typeof value === "object" ? JSON.stringify(value, null, 2) : value.toString()
                                }
                              />
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
                <AccordionHero
                  selectedStrategy={selectedStrategy}
                  strategyArgs={
                    routerSettings &&
                    routerSettings["routing_strategy_args"] &&
                    Object.keys(routerSettings["routing_strategy_args"]).length > 0
                      ? routerSettings["routing_strategy_args"]
                      : defaultLowestLatencyArgs // default value when keys length is 0
                  }
                  paramExplanation={paramExplanation}
                />
              </Card>
              <Col>
                <Button className="mt-2" onClick={() => handleSaveChanges(routerSettings)}>
                  Save Changes
                </Button>
              </Col>
            </Grid>
          </TabPanel>
          <TabPanel>
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
          </TabPanel>
          <TabPanel>
            <Card>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableHeaderCell>Setting</TableHeaderCell>
                    <TableHeaderCell>Value</TableHeaderCell>
                    <TableHeaderCell>Status</TableHeaderCell>
                    <TableHeaderCell>Action</TableHeaderCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {generalSettings
                    .filter((value) => value.field_type !== "TypedDictionary")
                    .map((value, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          <Text>{value.field_name}</Text>
                          <p
                            style={{
                              fontSize: "0.65rem",
                              color: "#808080",
                              fontStyle: "italic",
                            }}
                            className="mt-1"
                          >
                            {value.field_description}
                          </p>
                        </TableCell>
                        <TableCell>
                          {value.field_type == "Integer" ? (
                            <InputNumber
                              step={1}
                              value={value.field_value}
                              onChange={(newValue) => handleInputChange(value.field_name, newValue)} // Handle value change
                            />
                          ) : null}
                        </TableCell>
                        <TableCell>
                          {value.stored_in_db == true ? (
                            <Badge icon={CheckCircleIcon} className="text-white">
                              In DB
                            </Badge>
                          ) : value.stored_in_db == false ? (
                            <Badge className="text-gray bg-white outline">In Config</Badge>
                          ) : (
                            <Badge className="text-gray bg-white outline">Not Set</Badge>
                          )}
                        </TableCell>
                        <TableCell>
                          <Button onClick={() => handleUpdateField(value.field_name, index)}>Update</Button>
                          <Icon icon={TrashIcon} color="red" onClick={() => handleResetField(value.field_name, index)}>
                            Reset
                          </Icon>
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default GeneralSettings;
