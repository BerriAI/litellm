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
import { getBudgetSettings } from "../networking";
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
import AddFallbacks from "../add_fallbacks";
import openai from "openai";
import Paragraph from "antd/es/skeleton/Paragraph";

interface BudgetSettingsPageProps {
  accessToken: string | null;
}

interface budgetSettingsItem {
  field_name: string;
  field_type: string;
  field_value: any;
  field_description: string;
}

const BudgetSettings: React.FC<BudgetSettingsPageProps> = ({ accessToken }) => {
  const [budgetSettings, setBudgetSettings] = useState<budgetSettingsItem[]>(
    []
  );
  useEffect(() => {
    if (!accessToken) {
      return;
    }
    getBudgetSettings(accessToken).then((data) => {
      console.log("budget settings", data);
      let budget_settings = data.budget_settings;
      setBudgetSettings(budget_settings);
    });
  }, [accessToken]);

  const handleInputChange = (fieldName: string, newValue: any) => {
    // Update the value in the state
    const updatedSettings = budgetSettings.map((setting) =>
      setting.field_name === fieldName
        ? { ...setting, field_value: newValue }
        : setting
    );
    setBudgetSettings(updatedSettings);
  };

  const handleUpdateField = (fieldName: string, idx: number) => {
    if (!accessToken) {
      return;
    }

    let fieldValue = budgetSettings[idx].field_value;

    if (fieldValue == null || fieldValue == undefined) {
      return;
    }
    try {
      const updatedSettings = budgetSettings.map((setting) =>
        setting.field_name === fieldName
          ? { ...setting, stored_in_db: true }
          : setting
      );
      setBudgetSettings(updatedSettings);
    } catch (error) {
      // do something
    }
  };

  const handleResetField = (fieldName: string, idx: number) => {
    if (!accessToken) {
      return;
    }

    try {
      const updatedSettings = budgetSettings.map((setting) =>
        setting.field_name === fieldName
          ? { ...setting, stored_in_db: null, field_value: null }
          : setting
      );
      setBudgetSettings(updatedSettings);
    } catch (error) {
      // do something
    }
  };

  return (
    <div className="w-full mx-4">
      <Card>
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>Setting</TableHeaderCell>
              <TableHeaderCell>Value</TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {budgetSettings.map((value, index) => (
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
                      onChange={(newValue) =>
                        handleInputChange(value.field_name, newValue)
                      } // Handle value change
                    />
                  ) : null}
                </TableCell>
                <TableCell>
                  <Button
                    onClick={() => handleUpdateField(value.field_name, index)}
                  >
                    Update
                  </Button>
                  <Icon
                    icon={TrashIcon}
                    color="red"
                    onClick={() => handleResetField(value.field_name, index)}
                  >
                    Reset
                  </Icon>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
    </div>
  );
};

export default BudgetSettings;
