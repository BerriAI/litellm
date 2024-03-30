import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Subtitle,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableCell,
  TableBody,
  Metric,
  Text,
  Grid,
} from "@tremor/react";
import { modelInfoCall, userGetRequesedtModelsCall, modelMetricsCall } from "./networking";
import { BarChart } from "@tremor/react";
import { Badge, BadgeDelta, Button } from "@tremor/react";
import RequestAccess from "./request_model_access";
import { Typography } from "antd";

interface ModelDashboardProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
}

const ModelDashboard: React.FC<ModelDashboardProps> = ({
  accessToken,
  token,
  userRole,
  userID,
}) => {
  const [modelData, setModelData] = useState<any>({ data: [] });
  const [modelMetrics, setModelMetrics] = useState<any[]>([]);
  const [pendingRequests, setPendingRequests] = useState<any[]>([]);

  useEffect(() => {
    if (!accessToken || !token || !userRole || !userID) {
      return;
    }
    const fetchData = async () => {
      try {
        // Replace with your actual API call for model data
        const modelDataResponse = await modelInfoCall(
          accessToken,
          userID,
          userRole
        );
        console.log("Model data response:", modelDataResponse.data);
        setModelData(modelDataResponse);

        const modelMetricsResponse = await modelMetricsCall(
          accessToken,
          userID,
          userRole
        );

        console.log("Model metrics response:", modelMetricsResponse);
        setModelMetrics(modelMetricsResponse);

        // if userRole is Admin, show the pending requests
        if (userRole === "Admin" && accessToken) {
          const user_requests = await userGetRequesedtModelsCall(accessToken);
          console.log("Pending Requests:", pendingRequests);
          setPendingRequests(user_requests.requests || []);
        }
      } catch (error) {
        console.error("There was an error fetching the model data", error);
      }
    };

    if (accessToken && token && userRole && userID) {
      fetchData();
    }
  }, [accessToken, token, userRole, userID]);

  if (!modelData) {
    return <div>Loading...</div>;
  }

  if (!accessToken || !token || !userRole || !userID) {
    return <div>Loading...</div>;
  }
  let all_models_on_proxy: any[] = [];

  // loop through model data and edit each row
  for (let i = 0; i < modelData.data.length; i++) {
    let curr_model = modelData.data[i];
    let litellm_model_name = curr_model?.litellm_params?.mode
    let model_info = curr_model?.model_info;

    let defaultProvider = "openai";
    let provider = "";
    let input_cost = "Undefined";
    let output_cost = "Undefined";
    let max_tokens = "Undefined";

    // Check if litellm_model_name is null or undefined
    if (litellm_model_name) {
      // Split litellm_model_name based on "/"
      let splitModel = litellm_model_name.split("/");

      // Get the first element in the split
      let firstElement = splitModel[0];

      // If there is only one element, default provider to openai
      provider = splitModel.length === 1 ? defaultProvider : firstElement;
    } else {
      // litellm_model_name is null or undefined, default provider to openai
      provider = defaultProvider;
    }

    if (model_info) {
      input_cost = model_info?.input_cost_per_token;
      output_cost = model_info?.output_cost_per_token;
      max_tokens = model_info?.max_tokens;
    }
    modelData.data[i].provider = provider;
    modelData.data[i].input_cost = input_cost;
    modelData.data[i].output_cost = output_cost;
    modelData.data[i].max_tokens = max_tokens;
    modelData.data[i].api_base = curr_model?.litellm_params?.api_base;

    all_models_on_proxy.push(curr_model.model_name);

    console.log(modelData.data[i]);
  }
  // when users click request access show pop up to allow them to request access

  if (userRole && userRole == "Admin Viewer") {
    const { Title, Paragraph } = Typography;
    return (
      <div>
        <Title level={1}>Access Denied</Title>
        <Paragraph>
          Ask your proxy admin for access to view all models
        </Paragraph>
      </div>
    );
  }

  return (
    <div style={{ width: "100%" }}>
      <Grid className="gap-2 p-8 h-[75vh] w-full mt-2">
        <Card>
          <Table className="mt-5">
            <TableHead>
              <TableRow>

                  <TableHeaderCell>Model Name </TableHeaderCell>

                <TableHeaderCell>
                  Provider
                </TableHeaderCell>
                {
                  userRole === "Admin" && (
                    <TableHeaderCell>
                      API Base
                    </TableHeaderCell>
                  )
                }
                <TableHeaderCell>
                  Access
                </TableHeaderCell>
                <TableHeaderCell>Input Price per token ($)</TableHeaderCell>
                <TableHeaderCell>Output Price per token ($)</TableHeaderCell>
                <TableHeaderCell>Max Tokens</TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {modelData.data.map((model: any) => (
                <TableRow key={model.model_name}>
                  <TableCell>
                    <Text>{model.model_name}</Text>
                  </TableCell>
                  <TableCell>{model.provider}</TableCell>
                  {
                    userRole === "Admin" && (
                      <TableCell>{model.api_base}</TableCell>
                    )
                  }

                  <TableCell>
                    {model.user_access ? (
                      <Badge color={"green"}>Yes</Badge>
                    ) : (
                      <RequestAccess
                        userModels={all_models_on_proxy}
                        accessToken={accessToken}
                        userID={userID}
                      ></RequestAccess>
                    )}
                  </TableCell>

                  <TableCell>{model.input_cost}</TableCell>
                  <TableCell>{model.output_cost}</TableCell>
                  <TableCell>{model.max_tokens}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
        <Card>
          <Title>Model Statistics (Number Requests)</Title>
              <BarChart
                data={modelMetrics}
                index="model"
                categories={["num_requests"]}
                colors={["blue"]}
                yAxisWidth={400}
                layout="vertical"
                tickGap={5}
              />
        </Card>
        <Card>
          <Title>Model Statistics (Latency)</Title>
              <BarChart
                data={modelMetrics}
                index="model"
                categories={["avg_latency_seconds"]}
                colors={["red"]}
                yAxisWidth={400}
                layout="vertical"
                tickGap={5}
              />
        </Card>
        {/* {userRole === "Admin" &&
        pendingRequests &&
        pendingRequests.length > 0 ? (
          <Card>
            <Table>
              <TableHead>
                <Title>Pending Requests</Title>
                <TableRow>
                  <TableCell>
                    <Title>User ID</Title>
                  </TableCell>
                  <TableCell>
                    <Title>Requested Models</Title>
                  </TableCell>
                  <TableCell>
                    <Title>Justification</Title>
                  </TableCell>
                  <TableCell>
                    <Title>Justification</Title>
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {pendingRequests.map((request: any) => (
                  <TableRow key={request.request_id}>
                    <TableCell>
                      <p>{request.user_id}</p>
                    </TableCell>
                    <TableCell>
                      <p>{request.models[0]}</p>
                    </TableCell>
                    <TableCell>
                      <p>{request.justification}</p>
                    </TableCell>
                    <TableCell>
                      <p>{request.user_id}</p>
                    </TableCell>
                    <Button>Approve</Button>
                    <Button variant="secondary" className="ml-2">
                      Deny
                    </Button>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        ) : null} */}
      </Grid>
    </div>
  );
};

export default ModelDashboard;
