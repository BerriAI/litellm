import React, { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CheckCircle, Trash2 } from "lucide-react";
import {
  getGeneralSettingsCall,
  updateConfigFieldSetting,
  deleteConfigFieldSetting,
} from "./networking";
import RouterSettings from "./router_settings";
import Fallbacks from "./Settings/RouterSettings/Fallbacks/Fallbacks";

interface GeneralSettingsPageProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  modelData: any;
}

interface generalSettingsItem {
  field_name: string;
  field_type: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  field_value: any;
  field_description: string;
  stored_in_db: boolean | null;
}

const GeneralSettings: React.FC<GeneralSettingsPageProps> = ({
  accessToken,
  userRole,
  userID,
  modelData,
}) => {
  const [generalSettings, setGeneralSettings] = useState<
    generalSettingsItem[]
  >([]);

  useEffect(() => {
    if (!accessToken) return;
    getGeneralSettingsCall(accessToken).then((data) => {
      setGeneralSettings(data);
    });
  }, [accessToken]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleInputChange = (fieldName: string, newValue: any) => {
    const updatedSettings = generalSettings.map((setting) =>
      setting.field_name === fieldName
        ? { ...setting, field_value: newValue }
        : setting,
    );
    setGeneralSettings(updatedSettings);
  };

  const handleUpdateField = (fieldName: string, idx: number) => {
    if (!accessToken) return;
    const fieldValue = generalSettings[idx].field_value;
    if (fieldValue == null || fieldValue == undefined) return;
    try {
      updateConfigFieldSetting(accessToken, fieldName, fieldValue);
      const updatedSettings = generalSettings.map((setting) =>
        setting.field_name === fieldName
          ? { ...setting, stored_in_db: true }
          : setting,
      );
      setGeneralSettings(updatedSettings);
    } catch {
      // ignore
    }
  };

  const handleResetField = (fieldName: string) => {
    if (!accessToken) return;
    try {
      deleteConfigFieldSetting(accessToken, fieldName);
      const updatedSettings = generalSettings.map((setting) =>
        setting.field_name === fieldName
          ? { ...setting, stored_in_db: null, field_value: null }
          : setting,
      );
      setGeneralSettings(updatedSettings);
    } catch {
      // ignore
    }
  };

  if (!accessToken) return null;

  return (
    <div className="w-full">
      <Tabs defaultValue="loadbalancing" className="h-[75vh] w-full">
        <TabsList className="px-8 pt-4 mx-8">
          <TabsTrigger value="loadbalancing">Loadbalancing</TabsTrigger>
          <TabsTrigger value="fallbacks">Fallbacks</TabsTrigger>
          <TabsTrigger value="general">General</TabsTrigger>
        </TabsList>
        <div className="px-8 py-6">
          <TabsContent value="loadbalancing">
            <RouterSettings
              accessToken={accessToken}
              userRole={userRole}
              userID={userID}
              modelData={modelData}
            />
          </TabsContent>
          <TabsContent value="fallbacks">
            <Fallbacks
              accessToken={accessToken}
              userRole={userRole}
              userID={userID}
              modelData={modelData}
            />
          </TabsContent>
          <TabsContent value="general">
            <Card className="p-6">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Setting</TableHead>
                    <TableHead>Value</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Action</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {generalSettings
                    .filter((value) => value.field_type !== "TypedDictionary")
                    .map((value, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          <p>{value.field_name}</p>
                          <p className="text-[0.65rem] text-muted-foreground italic mt-1">
                            {value.field_description}
                          </p>
                        </TableCell>
                        <TableCell>
                          {value.field_type === "Integer" ? (
                            <Input
                              type="number"
                              step={1}
                              value={value.field_value ?? ""}
                              onChange={(e) =>
                                handleInputChange(
                                  value.field_name,
                                  e.target.value === ""
                                    ? null
                                    : Number(e.target.value),
                                )
                              }
                            />
                          ) : value.field_type === "Boolean" ? (
                            <Switch
                              checked={
                                value.field_value === true ||
                                value.field_value === "true"
                              }
                              onCheckedChange={(checked) =>
                                handleInputChange(value.field_name, checked)
                              }
                            />
                          ) : null}
                        </TableCell>
                        <TableCell>
                          {value.stored_in_db === true ? (
                            <Badge className="gap-1">
                              <CheckCircle size={12} />
                              In DB
                            </Badge>
                          ) : value.stored_in_db === false ? (
                            <Badge variant="outline">In Config</Badge>
                          ) : (
                            <Badge variant="outline">Not Set</Badge>
                          )}
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Button
                              size="sm"
                              onClick={() =>
                                handleUpdateField(value.field_name, index)
                              }
                            >
                              Update
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 text-destructive"
                              onClick={() => handleResetField(value.field_name)}
                              aria-label="Reset"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </Card>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
};

export default GeneralSettings;
