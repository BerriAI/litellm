import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Table,
  TableHead,
  TableRow,
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
import {
  getCallbacksCall,
  setCallbacksCall,
  getRouterSettingsCall,
} from "./networking";
import NotificationsManager from "./molecules/notifications_manager";

interface RouterSettingsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  modelData: any;
}

interface routingStrategyArgs {
  ttl?: number;
  lowest_latency_buffer?: number;
}

interface AccordionHeroProps {
  selectedStrategy: string | null;
  strategyArgs: routingStrategyArgs;
  paramExplanation: { [key: string]: string };
}

const defaultLowestLatencyArgs: routingStrategyArgs = {
  ttl: 3600,
  lowest_latency_buffer: 0,
};

const AccordionHero: React.FC<AccordionHeroProps> = ({ selectedStrategy, strategyArgs, paramExplanation }) => (
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

const RouterSettings: React.FC<RouterSettingsProps> = ({ accessToken, userRole, userID, modelData }) => {
  const [routerSettings, setRouterSettings] = useState<{ [key: string]: any }>({});
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);
  const [availableRoutingStrategies, setAvailableRoutingStrategies] = useState<string[]>([]);
  const [routerFieldsMetadata, setRouterFieldsMetadata] = useState<{ [key: string]: any }>({});

  // Param explanations for routing strategy args (these are dynamic based on strategy)
  let paramExplanation: { [key: string]: string } = {
    ttl: "Sliding window to look back over when calculating the average latency of a deployment. Default - 1 hour (in seconds).",
    lowest_latency_buffer:
      "Shuffle between deployments within this % of the lowest latency. Default - 0 (i.e. always pick lowest latency).",
  };

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
    getRouterSettingsCall(accessToken).then((data) => {
      console.log("router settings from API", data);
      if (data.fields) {
        // Build metadata map for easy lookup
        const fieldsMap: { [key: string]: any } = {};
        data.fields.forEach((field: any) => {
          fieldsMap[field.field_name] = {
            ui_field_name: field.ui_field_name,
            field_description: field.field_description,
            options: field.options,
          };
        });
        setRouterFieldsMetadata(fieldsMap);
        
        // Extract routing strategies from the routing_strategy field's options
        const routingStrategyField = data.fields.find(
          (field: any) => field.field_name === "routing_strategy"
        );
        if (routingStrategyField?.options) {
          setAvailableRoutingStrategies(routingStrategyField.options);
        }
      }
    });
  }, [accessToken, userRole, userID]);

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
                    <Text>{routerFieldsMetadata[param]?.ui_field_name || param}</Text>
                    <p
                      style={{
                        fontSize: "0.65rem",
                        color: "#808080",
                        fontStyle: "italic",
                      }}
                      className="mt-1"
                    >
                      {routerFieldsMetadata[param]?.field_description || ""}
                    </p>
                  </TableCell>
                  <TableCell>
                    {param == "routing_strategy" ? (
                      <Select2
                        defaultValue={value}
                        className="w-full max-w-md"
                        onValueChange={setSelectedStrategy}
                      >
                        {availableRoutingStrategies.map((strategy) => (
                          <SelectItem key={strategy} value={strategy}>
                            {strategy}
                          </SelectItem>
                        ))}
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
              : defaultLowestLatencyArgs
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
  );
};

export default RouterSettings;

