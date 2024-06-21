import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  BarChart,
  Subtitle,
  Grid,
  Col,
  Select,
  SelectItem,
  DateRangePicker,
  DateRangePickerValue,
  MultiSelect,
  MultiSelectItem,
} from "@tremor/react";

import {
    adminGlobalCacheActivity,
} from "./networking";

const formatDateWithoutTZ = (date: Date | undefined) => {
    if (!date) return undefined;
    return date.toISOString().split('T')[0];
  };

interface CachePageProps {
    accessToken: string | null;
    token: string | null;
    userRole: string | null;
    userID: string | null;
    premiumUser: boolean;
}


const CacheDashboard: React.FC<CachePageProps> = ({
  accessToken,
  token,
  userRole,
  userID,
  premiumUser,
}) => {
  const [filteredData, setFilteredData] = useState([]);
  const [selectedApiKeys, setSelectedApiKeys] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [data, setData] = useState([]);

  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), 
    to: new Date(),
  });


  useEffect(() => {
    if (!accessToken || !dateValue) {
      return;
    }
    const fetchData = async () => {
      const response = await adminGlobalCacheActivity(accessToken, formatDateWithoutTZ(dateValue.from), formatDateWithoutTZ(dateValue.to));
      setData(response);
    };
    fetchData();
  }, [accessToken]);

  const uniqueApiKeys = [...new Set(data.map((item) => item.api_key))];
  const uniqueModels = [...new Set(data.map((item) => item.model))];
  const uniqueCallTypes = [...new Set(data.map((item) => item.call_type))];

  useEffect(() => {
    console.log("DATA IN CACHE DASHBOARD", data);
    let newData = data;
    if (selectedApiKeys.length > 0) {
      newData = newData.filter((item) => selectedApiKeys.includes(item.api_key));
    }

    if (selectedModels.length > 0) {
      newData = newData.filter((item) => selectedModels.includes(item.model));
    }

    /* 
    Data looks like this 
    [{"api_key":"147dba2181f28914eea90eb484926c293cdcf7f5b5c9c3dd6a004d9e0f9fdb21","call_type":"acompletion","model":"llama3-8b-8192","total_rows":13,"cache_hit_true_rows":0},
    {"api_key":"8c23f021d0535c2e59abb7d83d0e03ccfb8db1b90e231ff082949d95df419e86","call_type":"None","model":"chatgpt-v-2","total_rows":1,"cache_hit_true_rows":0},
    {"api_key":"88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b","call_type":"acompletion","model":"gpt-3.5-turbo","total_rows":19,"cache_hit_true_rows":0},
    {"api_key":"88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b","call_type":"aimage_generation","model":"","total_rows":3,"cache_hit_true_rows":0},
    {"api_key":"0ad4b3c03dcb6de0b5b8f761db798c6a8ae80be3fd1e2ea30c07ce6d5e3bf870","call_type":"None","model":"chatgpt-v-2","total_rows":1,"cache_hit_true_rows":0},
    {"api_key":"034224b36e9769bc50e2190634abc3f97cad789b17ca80ac43b82f46cd5579b3","call_type":"","model":"chatgpt-v-2","total_rows":1,"cache_hit_true_rows":0},
    {"api_key":"4f9c71cce0a2bb9a0b62ce6f0ebb3245b682702a8851d26932fa7e3b8ebfc755","call_type":"","model":"chatgpt-v-2","total_rows":1,"cache_hit_true_rows":0},
    */

    // What data we need for bar chat 
    // ui_data = [
    //     {
    //         name: "Call Type",
    //         Cache hit: 20,
    //         LLM API requests: 10,
    //     }
    // ]

    console.log("before processed data in cache dashboard", newData);

    const processedData = newData.reduce((acc, item) => {
        console.log("Processing item:", item);
        
        if (!item.call_type) {
          console.log("Item has no call_type:", item);
          item.call_type = "Unknown";
        }
    
        const existingItem = acc.find(i => i.name === item.call_type);
        if (existingItem) {
          existingItem["LLM API requests"] += (item.total_rows || 0) - (item.cache_hit_true_rows || 0);
          existingItem["Cache hit"] += item.cache_hit_true_rows || 0;
        } else {
          acc.push({
            name: item.call_type,
            "LLM API requests": (item.total_rows || 0) - (item.cache_hit_true_rows || 0),
            "Cache hit": item.cache_hit_true_rows || 0,
          });
        }
        return acc;
      }, []);
  
    setFilteredData(processedData);

    console.log("PROCESSED DATA IN CACHE DASHBOARD", processedData);

  }, [selectedApiKeys, selectedModels, dateValue, data]);

  return (
    <Card>
      <Title>API Activity Dashboard</Title>
      <Subtitle>Cache hits vs API requests broken down by call type</Subtitle>
      
      <Grid numItems={3} className="gap-4 mt-4">
        <Col>
          <MultiSelect
            placeholder="Select API Keys"
            value={selectedApiKeys}
            onValueChange={setSelectedApiKeys}
          >
            {uniqueApiKeys.map((key) => (
              <MultiSelectItem key={key} value={key}>
                {key}
              </MultiSelectItem>
            ))}
          </MultiSelect>
        </Col>
        <Col>
          <MultiSelect
            placeholder="Select Models"
            value={selectedModels}
            onValueChange={setSelectedModels}
          >
            {uniqueModels.map((model) => (
              <MultiSelectItem key={model} value={model}>
                {model}
              </MultiSelectItem>
            ))}
          </MultiSelect>
        </Col>
        <Col>
          <DateRangePicker
            value={dateValue}
            // onChange={setDateRange}
            selectPlaceholder="Select date range"
          />
        </Col>
      </Grid>

      <BarChart
        className="mt-6"
        data={filteredData}
        index="name"
        categories={["LLM API requests", "Cache hit"]}
        colors={["blue", "teal"]}
        yAxisWidth={48}
      />
    </Card>
  );
};

export default CacheDashboard;