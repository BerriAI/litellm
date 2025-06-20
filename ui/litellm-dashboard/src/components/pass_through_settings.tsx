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
import PassThroughInfoView from "./pass_through_info";
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
  id?: string
  path: string
  target: string
  headers: object
  include_subpath?: boolean
  cost_per_request?: number
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
  const [selectedEndpointId, setSelectedEndpointId] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
    getPassThroughEndpointsCall(accessToken).then((data) => {
      let general_settings = data["endpoints"];
      setGeneralSettings(general_settings);
    });
  }, [accessToken, userRole, userID]);

  const handleEndpointUpdated = () => {
    // Refresh the endpoints list when an endpoint is updated
    if (accessToken) {
      getPassThroughEndpointsCall(accessToken).then((data) => {
        let general_settings = data["endpoints"];
        setGeneralSettings(general_settings);
      });
    }
  };

  const handleResetField = (endpointId: string, idx: number) => {
    if (!accessToken) {
      return;
    }

    try {
      deletePassThroughEndpointsCall(accessToken, endpointId);
      
      const updatedSettings = generalSettings.filter((setting) => setting.id !== endpointId);
      setGeneralSettings(updatedSettings);

      message.success("Endpoint deleted successfully.");
    } catch (error) {
      // do something
    }
  };

  // Define columns for the DataTable
  const columns: ColumnDef<passThroughItem>[] = [
    {
      header: "ID",
      accessorKey: "id",
      cell: (info: any) => (
        <Text className="font-mono text-xs text-gray-500 truncate max-w-[120px]">
          {info.getValue()}
        </Text>
      ),
    },
    {
      header: "Path",
      accessorKey: "path",
      cell: (info: any) => (
        <div className="overflow-hidden">
          <Button 
            size="xs"
            variant="light"
            className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
            onClick={() => info.row.original.id && setSelectedEndpointId(info.row.original.id)}
          >
            {info.getValue()}
          </Button>
        </div>
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
      header: "Actions",
      id: "actions",
      cell: ({ row }) => (
        <div className="flex space-x-1">
          <Icon
            icon={PencilAltIcon}
            size="sm"
            onClick={() => row.original.id && setSelectedEndpointId(row.original.id)}
            title="Edit"
          />
          <Icon
            icon={TrashIcon}
            size="sm"
            onClick={() => handleResetField(row.original.id!, row.index)}
            title="Delete"
          />
        </div>
      ),
    },
  ];

  if (!accessToken) {
    return null;
  }

  // If a specific endpoint is selected, show the info view
  if (selectedEndpointId) {
    // Find the endpoint by ID to get the path for the info view
    const selectedEndpoint = generalSettings.find(endpoint => endpoint.id === selectedEndpointId);
    const endpointPath = selectedEndpoint?.path || selectedEndpointId;
    
    return (
      <PassThroughInfoView
        endpointPath={endpointPath}
        onClose={() => setSelectedEndpointId(null)}
        accessToken={accessToken}
        isAdmin={userRole === "Admin" || userRole === "admin"}
        onEndpointUpdated={handleEndpointUpdated}
      />
    );
  }

  return (
    <div>
        <div>
          <Title>Pass Through Endpoints</Title>
          <Text className="text-tremor-content">
            Configure and manage your pass-through endpoints
          </Text>
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
