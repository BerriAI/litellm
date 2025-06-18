import React, { useState, useEffect } from "react";
import {
  Badge,
  Metric,
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
  AccordionList,
  Icon,
  Title,
} from "@tremor/react";
import {
  getCallbacksCall,
  setCallbacksCall,
  getGeneralSettingsCall,
  deletePassThroughEndpointsCall,
  getPassThroughEndpointsCall,
  serviceHealthCheck,
  updateConfigFieldSetting,
  deleteConfigFieldSetting,
} from "./networking";
import {
  Modal,
  Form,
  Input,
  Select,
  Button as Button2,
  message,
  InputNumber,
} from "antd";
import {
  InformationCircleIcon,
  PencilAltIcon,
  PencilIcon,
  StatusOnlineIcon,
  TrashIcon,
  RefreshIcon,
  CheckCircleIcon,
  XCircleIcon,
  QuestionMarkCircleIcon,
} from "@heroicons/react/outline";
import AddFallbacks from "./add_fallbacks";
import AddPassThroughEndpoint from "./add_pass_through";
import openai from "openai";
import Paragraph from "antd/es/skeleton/Paragraph";
import { DataTable } from "./view_logs/table";
import { ColumnDef } from "@tanstack/react-table";
import { Eye, EyeOff } from "lucide-react";

interface GeneralSettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  modelData: any;
}

interface routingStrategyArgs {
  ttl?: number;
  lowest_latency_buffer?: number;
}

interface nestedFieldItem {
  field_name: string;
  field_type: string; 
  field_value: any; 
  field_description: string; 
  stored_in_db: boolean | null;
}

export interface passThroughItem {
  path: string
  target: string
  headers: object
  include_subpath?: boolean
  input_cost_per_request?: number
}

// Password field component for headers
const PasswordField: React.FC<{ value: object }> = ({ value }) => {
  const [showPassword, setShowPassword] = useState(false);
  const headerString = JSON.stringify(value);
  
  return (
    <div className="flex items-center space-x-2">
      <span className="font-mono text-xs">
        {showPassword ? headerString : "••••••••"}
      </span>
      <button
        onClick={() => setShowPassword(!showPassword)}
        className="p-1 hover:bg-gray-100 rounded"
        type="button"
      >
        {showPassword ? (
          <EyeOff className="w-4 h-4 text-gray-500" />
        ) : (
          <Eye className="w-4 h-4 text-gray-500" />
        )}
      </button>
    </div>
  );
};

const PassThroughSettings: React.FC<GeneralSettingsPageProps> = ({
  accessToken,
  userRole,
  userID,
  modelData,
}) => {
  const [generalSettings, setGeneralSettings] = useState<passThroughItem[]>(
    []
  );

  useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
    getPassThroughEndpointsCall(accessToken).then((data) => {
      let general_settings = data["endpoints"];
      setGeneralSettings(general_settings);
    });
  }, [accessToken, userRole, userID]);

  const handleResetField = (fieldName: string, idx: number) => {
    if (!accessToken) {
      return;
    }

    try {
      deletePassThroughEndpointsCall(accessToken, fieldName);
      
      const updatedSettings = generalSettings.filter((setting) => setting.path !== fieldName);
      setGeneralSettings(updatedSettings);

      message.success("Endpoint deleted successfully.");
    } catch (error) {
      // do something
    }
  };

  // Define columns for the DataTable
  const columns: ColumnDef<passThroughItem>[] = [
    {
      header: "Path",
      accessorKey: "path",
      cell: (info: any) => (
        <Text className="font-mono">{info.getValue()}</Text>
      ),
    },
    {
      header: "Target",
      accessorKey: "target",
      cell: (info: any) => (
        <Text>{info.getValue()}</Text>
      ),
    },
    {
      header: "Headers",
      accessorKey: "headers",
      cell: (info: any) => (
        <PasswordField value={info.getValue() || {}} />
      ),
    },
    {
      header: "Action",
      id: "actions",
      cell: ({ row }) => (
        <Icon
          icon={TrashIcon}
          color="red"
          className="cursor-pointer"
          onClick={() => handleResetField(row.original.path, row.index)}
        >
          Delete
        </Icon>
      ),
    },
  ];

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full h-[75vh] p-6">
      <div className="mb-2 mt-4">
        <div>
          <Title>Pass Through Endpoints</Title>
          <Text className="text-tremor-content">
            Configure and manage your pass-through endpoints
          </Text>
        </div>
      </div>
      
      <AddPassThroughEndpoint 
        accessToken={accessToken} 
        setPassThroughItems={setGeneralSettings} 
        passThroughItems={generalSettings}
      />
      
      <DataTable
        data={generalSettings}
        columns={columns}
        renderSubComponent={() => <div></div>}
        getRowCanExpand={() => false}
        isLoading={false}
        noDataMessage="No pass-through endpoints configured"
      />
    </div>
  );
};

export default PassThroughSettings;
