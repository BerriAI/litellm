import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Subtitle,
  Table,
  TableHead,
  TableRow,
  Badge,
  TableHeaderCell,
  TableCell,
  TableBody,
  Metric,
  Text,
  Grid,
  Button,
  TextInput,
  Col,
} from "@tremor/react";
import { TabPanel, TabPanels, TabGroup, TabList, Tab, Icon } from "@tremor/react";
import { getCallbacksCall, setCallbacksCall, serviceHealthCheck } from "./networking";
import { Modal, Form, Input, Select, Button as Button2, message } from "antd";
import { InformationCircleIcon, PencilAltIcon, PencilIcon, StatusOnlineIcon, TrashIcon, RefreshIcon } from "@heroicons/react/outline";
import StaticGenerationSearchParamsBailoutProvider from "next/dist/client/components/static-generation-searchparams-bailout-provider";
import AddFallbacks from "./add_fallbacks"
import openai from "openai";

interface GeneralSettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  modelData: any
}

async function testFallbackModelResponse(
  selectedModel: string,
  accessToken: string
) {
  // base url should be the current base_url
  const isLocal = process.env.NODE_ENV === "development";
  console.log("isLocal:", isLocal);
  const proxyBaseUrl = isLocal
    ? "http://localhost:4000"
    : window.location.origin;
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
      mock_testing_fallbacks: true
    });

    message.success(
      <span>
        Test model=<strong>{selectedModel}</strong>, received model=<strong>{response.model}</strong>. 
        See <a href="#" onClick={() => window.open('https://docs.litellm.ai/docs/proxy/reliability', '_blank')} style={{ textDecoration: 'underline', color: 'blue' }}>curl</a>
      </span>
    );
  } catch (error) {
    message.error(`Error occurred while generating model response. Please try again. Error: ${error}`, 20);
  }
}

const GeneralSettings: React.FC<GeneralSettingsPageProps> = ({
  accessToken,
  userRole,
  userID,
  modelData
}) => {
  const [routerSettings, setRouterSettings] = useState<{ [key: string]: any }>({});
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [selectedCallback, setSelectedCallback] = useState<string | null>(null);

  let paramExplanation: { [key: string]: string } = {
    "routing_strategy_args": "(dict) Arguments to pass to the routing strategy",
    "routing_strategy": "(string) Routing strategy to use",
    "allowed_fails": "(int) Number of times a deployment can fail before being added to cooldown",
    "cooldown_time": "(int) time in seconds to cooldown a deployment after failure",
    "num_retries": "(int) Number of retries for failed requests. Defaults to 0.",
    "timeout": "(float) Timeout for requests. Defaults to None.",
    "retry_after": "(int) Minimum time to wait before retrying a failed request",
  }

  useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
    getCallbacksCall(accessToken, userID, userRole).then((data) => {
      console.log("callbacks", data);
      let router_settings = data.router_settings;
      setRouterSettings(router_settings);
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

    console.log(`received key: ${key}`)
    console.log(`routerSettings['fallbacks']: ${routerSettings['fallbacks']}`)

    routerSettings["fallbacks"].map((dict: { [key: string]: any }) => {
      // Check if the dictionary has the specified key and delete it if present
      if (key in dict) {
        delete dict[key];
      }
      return dict; // Return the updated dictionary
    });

    const payload = {
      router_settings: routerSettings
    };

    try {
      await setCallbacksCall(accessToken, payload);
      setRouterSettings({ ...routerSettings });
      message.success("Router settings updated successfully");
    } catch (error) {
      message.error("Failed to update router settings: " + error, 20);
    }
  }

  const handleSaveChanges = (router_settings: any) => {
    if (!accessToken) {
      return;
    }

    console.log("router_settings", router_settings);

    const updatedVariables = Object.fromEntries(
      Object.entries(router_settings).map(([key, value]) => {
        if (key !== 'routing_strategy_args') {
          return [key, (document.querySelector(`input[name="${key}"]`) as HTMLInputElement)?.value || value];
        }
        return null;
      }).filter(entry => entry !== null) as Iterable<[string, unknown]>
    );
    console.log("updatedVariables", updatedVariables);

    const payload = {
      router_settings: updatedVariables
    };

    try {
      setCallbacksCall(accessToken, payload);
    } catch (error) {
      message.error("Failed to update router settings: " + error, 20);
    }

    message.success("router settings updated successfully");
  };

  

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full mx-4">
      <TabGroup className="gap-2 p-8 h-[75vh] w-full mt-2">
        <TabList variant="line" defaultValue="1">
          <Tab value="1">General Settings</Tab>
          <Tab value="2">Fallbacks</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
      <Grid numItems={1} className="gap-2 p-8 w-full mt-2">
      <Title>Router Settings</Title>
        <Card >
          <Table>
            <TableHead>
              <TableRow>
                <TableHeaderCell>Setting</TableHeaderCell>
                <TableHeaderCell>Value</TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(routerSettings).filter(([param, value]) => param != "fallbacks" && param != "context_window_fallbacks").map(([param, value]) => (
                <TableRow key={param}>
                  <TableCell>
                    <Text>{param}</Text>
                    <p style={{fontSize: '0.65rem', color: '#808080', fontStyle: 'italic'}} className="mt-1">{paramExplanation[param]}</p>
                  </TableCell>
                  <TableCell>
                    <TextInput
                      name={param}
                      defaultValue={
                        typeof value === 'object' ? JSON.stringify(value, null, 2) : value.toString()
                      }
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
        </Table>
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
          {
            routerSettings["fallbacks"] &&
            routerSettings["fallbacks"].map((item: Object, index: number) =>
              Object.entries(item).map(([key, value]) => (
                <TableRow key={index.toString() + key}>
                  <TableCell>{key}</TableCell>
                  <TableCell>{Array.isArray(value) ? value.join(', ') : value}</TableCell>
                  <TableCell>
                    <Button onClick={() => testFallbackModelResponse(key, accessToken)}>
                      Test Fallback
                    </Button>
                  </TableCell>
                  <TableCell>
                    <Icon
                        icon={TrashIcon}
                        size="sm"
                        onClick={() => deleteFallbacks(key)}
                      />
                  </TableCell>
                </TableRow>
              ))
            )
          }
        </TableBody>
      </Table>
      <AddFallbacks models={modelData?.data ? modelData.data.map((data: any) => data.model_name) : []} accessToken={accessToken} routerSettings={routerSettings} setRouterSettings={setRouterSettings}/>
      </TabPanel>
      </TabPanels>
    </TabGroup>
    </div>
  );
};

export default GeneralSettings;