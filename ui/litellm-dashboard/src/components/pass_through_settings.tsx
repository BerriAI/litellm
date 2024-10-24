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
  Select as Select2,
  SelectItem,
  Col,
  Accordion,
  AccordionBody,
  AccordionHeader,
  AccordionList,
} from "@tremor/react";
import {
  TabPanel,
  TabPanels,
  TabGroup,
  TabList,
  Tab,
  Icon,
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
import StaticGenerationSearchParamsBailoutProvider from "next/dist/client/components/static-generation-searchparams-bailout-provider";
import AddFallbacks from "./add_fallbacks";
import AddPassThroughEndpoint from "./add_pass_through";
import openai from "openai";
import Paragraph from "antd/es/skeleton/Paragraph";
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
}




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
      // update value in state

      const updatedSettings = generalSettings.filter((setting) => setting.path !== fieldName);
      setGeneralSettings(updatedSettings);

      message.success("Endpoint deleted successfully.");

    } catch (error) {
      // do something
    }
  };


  if (!accessToken) {
    return null;
  }



  return (
    <div className="w-full mx-4">
      <TabGroup className="gap-2 p-8 h-[75vh] w-full mt-2">
        <Card>
          <Table>
            <TableHead>
              <TableRow>
                <TableHeaderCell>Path</TableHeaderCell>
                <TableHeaderCell>Target</TableHeaderCell>
                <TableHeaderCell>Headers</TableHeaderCell>
                <TableHeaderCell>Action</TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
            {generalSettings.map((value, index) => (
                    <TableRow key={index}>
                      <TableCell>
                        <Text>{value.path}</Text>
                      </TableCell>
                      <TableCell>
                        {
                          value.target
                        }
                      </TableCell>
                      <TableCell>
                        {
                          JSON.stringify(value.headers)
                        }
                      </TableCell>
                      <TableCell>
                        <Icon
                          icon={TrashIcon}
                          color="red"
                          onClick={() =>
                            handleResetField(value.path, index)
                          }
                        >
                          Reset
                        </Icon>
                      </TableCell>
                    </TableRow>
                  ))}
            </TableBody>
          </Table>
          <AddPassThroughEndpoint accessToken={accessToken} setPassThroughItems={setGeneralSettings} passThroughItems={generalSettings}/>
        </Card>
      </TabGroup>
    </div>
  );
};

export default PassThroughSettings;
