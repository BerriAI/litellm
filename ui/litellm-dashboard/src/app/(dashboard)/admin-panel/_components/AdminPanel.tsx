/**
 * Allow proxy admin to add other people to view global spend
 * Use this to avoid sharing master key with others
 */
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Info, TriangleAlert } from "lucide-react";
import React, { useEffect, useState } from "react";
import { useBaseUrl } from "@/components/constants";
import { toast } from "@/lib/toast";
import { addAllowedIP, deleteAllowedIP, getAllowedIPs, getSSOSettings } from "@/components/networking";
import SCIMConfig from "@/components/SCIM";
import LoggingSettings from "@/components/Settings/AdminSettings/LoggingSettings/LoggingSettings";
import SSOSettings from "@/components/Settings/AdminSettings/SSOSettings/SSOSettings";
import UISettings from "@/components/Settings/AdminSettings/UISettings/UISettings";
import UserBannerSettings from "@/components/Settings/AdminSettings/UserBannerSettings/UserBannerSettings";
import HashicorpVault from "@/components/Settings/AdminSettings/HashicorpVault/HashicorpVault";
import PluginSettings from "@/components/Settings/AdminSettings/PluginSettings/PluginSettings";
import SSOModals from "@/components/SSOModals";
import {
  emptySSOSettingsFormValues,
  useSSOSettingsForm,
  type SSOSettingsFormValues,
} from "@/components/Settings/AdminSettings/SSOSettings/Modals/BaseSSOSettingsForm";
import UIAccessControlForm from "@/components/UIAccessControlForm";
import { z } from "zod/v4";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Input } from "@/components/ui/input";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const allowedIPSchema = z.object({
  ip: z.string().min(1, "Please enter an IP address"),
});

type AllowedIPFormValues = z.infer<typeof allowedIPSchema>;

const AddAllowedIPForm = ({ onSubmit }: { onSubmit: (values: AllowedIPFormValues) => Promise<void> }) => {
  const form = useZodForm(allowedIPSchema, { defaultValues: { ip: "" } });

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <FieldGroup>
        <FormField control={form.control} name="ip">
          {({ ref, ...field }) => <Input ref={ref} placeholder="Enter IP address" {...field} />}
        </FormField>
        <div>
          <Button type="submit">Add IP Address</Button>
        </div>
      </FieldGroup>
    </form>
  );
};

interface AdminPanelProps {
  proxySettings?: any;
}

const AdminPanel: React.FC<AdminPanelProps> = ({ proxySettings }) => {
  const { premiumUser, accessToken, userId: userID } = useAuthorized();
  const form = useSSOSettingsForm("admin-panel");
  const [isAddSSOModalVisible, setIsAddSSOModalVisible] = useState(false);
  const [isInstructionsModalVisible, setIsInstructionsModalVisible] = useState(false);
  const [isAllowedIPModalVisible, setIsAllowedIPModalVisible] = useState(false);
  const [isAddIPModalVisible, setIsAddIPModalVisible] = useState(false);
  const [isDeleteIPModalVisible, setIsDeleteIPModalVisible] = useState(false);
  const [isUIAccessControlModalVisible, setIsUIAccessControlModalVisible] = useState(false);
  const [allowedIPs, setAllowedIPs] = useState<string[]>([]);
  const [ipToDelete, setIPToDelete] = useState<string | null>(null);
  const [ssoConfigured, setSsoConfigured] = useState<boolean>(false);

  const baseUrl = useBaseUrl();
  const all_ip_address_allowed = "All IP Addresses Allowed";

  let nonSssoUrl = baseUrl;
  nonSssoUrl += "/fallback/login";

  const checkSSOConfiguration = async () => {
    if (accessToken) {
      try {
        const ssoData = await getSSOSettings(accessToken);

        if (ssoData && ssoData.values) {
          const hasGoogleSSO = ssoData.values.google_client_id && ssoData.values.google_client_secret;
          const hasMicrosoftSSO = ssoData.values.microsoft_client_id && ssoData.values.microsoft_client_secret;
          const hasGenericSSO = ssoData.values.generic_client_id && ssoData.values.generic_client_secret;

          setSsoConfigured(hasGoogleSSO || hasMicrosoftSSO || hasGenericSSO);
        } else {
          setSsoConfigured(false);
        }
      } catch (error) {
        console.error("Error checking SSO configuration:", error);
        setSsoConfigured(false);
      }
    }
  };

  const handleShowAllowedIPs = async () => {
    try {
      if (premiumUser !== true) {
        toast.fromError("This feature is only available for premium users. Please upgrade your account.");
        return;
      }
      if (accessToken) {
        const data = await getAllowedIPs(accessToken);
        setAllowedIPs(data && data.length > 0 ? data : [all_ip_address_allowed]);
      } else {
        setAllowedIPs([all_ip_address_allowed]);
      }
    } catch (error) {
      console.error("Error fetching allowed IPs:", error);
      toast.fromError(`Failed to fetch allowed IPs ${error}`);
      setAllowedIPs([all_ip_address_allowed]);
    } finally {
      if (premiumUser === true) {
        setIsAllowedIPModalVisible(true);
      }
    }
  };

  const handleAddIP = async (values: { ip: string }) => {
    try {
      if (accessToken) {
        await addAllowedIP(accessToken, values.ip);
        // Fetch the updated list of IPs
        const updatedIPs = await getAllowedIPs(accessToken);
        setAllowedIPs(updatedIPs);
        toast.success("IP address added successfully");
      }
    } catch (error) {
      console.error("Error adding IP:", error);
      toast.fromError(`Failed to add IP address ${error}`);
    } finally {
      setIsAddIPModalVisible(false);
    }
  };

  const handleDeleteIP = async (ip: string) => {
    setIPToDelete(ip);
    setIsDeleteIPModalVisible(true);
  };

  const confirmDeleteIP = async () => {
    if (ipToDelete && accessToken) {
      try {
        await deleteAllowedIP(accessToken, ipToDelete);
        // Fetch the updated list of IPs
        const updatedIPs = await getAllowedIPs(accessToken);
        setAllowedIPs(updatedIPs.length > 0 ? updatedIPs : [all_ip_address_allowed]);
        toast.success("IP address deleted successfully");
      } catch (error) {
        console.error("Error deleting IP:", error);
        toast.fromError(`Failed to delete IP address ${error}`);
      } finally {
        setIsDeleteIPModalVisible(false);
        setIPToDelete(null);
      }
    }
  };

  const handleAddSSOOk = () => {
    setIsAddSSOModalVisible(false);
    form.reset(emptySSOSettingsFormValues);
    if (accessToken && premiumUser) {
      checkSSOConfiguration();
    }
  };

  const handleAddSSOCancel = () => {
    setIsAddSSOModalVisible(false);
    form.reset(emptySSOSettingsFormValues);
  };

  const handleShowInstructions = (formValues: SSOSettingsFormValues) => {
    setIsAddSSOModalVisible(false);
    setIsInstructionsModalVisible(true);
  };

  const handleInstructionsOk = () => {
    setIsInstructionsModalVisible(false);
    if (accessToken && premiumUser) {
      checkSSOConfiguration();
    }
  };

  const handleInstructionsCancel = () => {
    setIsInstructionsModalVisible(false);
    if (accessToken && premiumUser) {
      checkSSOConfiguration();
    }
  };

  useEffect(() => {
    checkSSOConfiguration();
  }, [accessToken, premiumUser, checkSSOConfiguration]);

  const handleUIAccessControlOk = () => {
    setIsUIAccessControlModalVisible(false);
  };

  const handleUIAccessControlCancel = () => {
    setIsUIAccessControlModalVisible(false);
  };

  const tabItems = [
    {
      key: "sso-settings",
      label: "SSO Settings",
      children: <SSOSettings />,
    },
    {
      key: "security-settings",
      label: "Security Settings",
      children: (
        <>
          <Card className="block p-6">
            <h3 className="mb-2 text-base font-semibold text-foreground">✨ Security Settings</h3>
            <Alert variant="warning">
              <TriangleAlert />
              <AlertTitle>SSO Configuration Deprecated</AlertTitle>
              <AlertDescription>
                Editing SSO Settings on this page is deprecated and will be removed in a future version. Please use the
                SSO Settings tab for SSO configuration.
              </AlertDescription>
            </Alert>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "1rem",
                marginTop: "1rem",
                marginLeft: "0.5rem",
              }}
            >
              <div>
                <Button style={{ width: "150px" }} onClick={() => setIsAddSSOModalVisible(true)}>
                  {ssoConfigured ? "Edit SSO Settings" : "Add SSO"}
                </Button>
              </div>
              <div>
                <Button style={{ width: "150px" }} onClick={handleShowAllowedIPs}>
                  Allowed IPs
                </Button>
              </div>
              <div>
                <Button
                  style={{ width: "150px" }}
                  onClick={() =>
                    premiumUser === true
                      ? setIsUIAccessControlModalVisible(true)
                      : toast.fromError("Only premium users can configure UI access control")
                  }
                >
                  UI Access Control
                </Button>
              </div>
            </div>
          </Card>

          <div className="flex justify-start mb-4">
            <SSOModals
              isAddSSOModalVisible={isAddSSOModalVisible}
              isInstructionsModalVisible={isInstructionsModalVisible}
              handleAddSSOOk={handleAddSSOOk}
              handleAddSSOCancel={handleAddSSOCancel}
              handleShowInstructions={handleShowInstructions}
              handleInstructionsOk={handleInstructionsOk}
              handleInstructionsCancel={handleInstructionsCancel}
              form={form}
              accessToken={accessToken}
              ssoConfigured={ssoConfigured}
            />
            <Dialog open={isAllowedIPModalVisible} onOpenChange={(open) => !open && setIsAllowedIPModalVisible(false)}>
              <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
                <DialogHeader>
                  <DialogTitle>Manage Allowed IP Addresses</DialogTitle>
                </DialogHeader>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>IP Address</TableHead>
                      <TableHead className="text-right">Action</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {allowedIPs.map((ip, index) => (
                      <TableRow key={index}>
                        <TableCell>{ip}</TableCell>
                        <TableCell className="text-right">
                          {ip !== all_ip_address_allowed && (
                            <Button onClick={() => handleDeleteIP(ip)} variant="destructive" size="sm">
                              Delete
                            </Button>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <DialogFooter>
                  <Button className="mx-1" onClick={() => setIsAddIPModalVisible(true)}>
                    Add IP Address
                  </Button>
                  <Button onClick={() => setIsAllowedIPModalVisible(false)}>Close</Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Dialog open={isAddIPModalVisible} onOpenChange={(open) => !open && setIsAddIPModalVisible(false)}>
              <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>Add Allowed IP Address</DialogTitle>
                </DialogHeader>
                <AddAllowedIPForm onSubmit={handleAddIP} />
              </DialogContent>
            </Dialog>

            <Dialog open={isDeleteIPModalVisible} onOpenChange={(open) => !open && setIsDeleteIPModalVisible(false)}>
              <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>Confirm Delete</DialogTitle>
                </DialogHeader>
                <span className="text-sm text-foreground">
                  Are you sure you want to delete the IP address: {ipToDelete}?
                </span>
                <DialogFooter>
                  <Button className="mx-1" onClick={() => confirmDeleteIP()}>
                    Yes
                  </Button>
                  <Button onClick={() => setIsDeleteIPModalVisible(false)}>Close</Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            {/* UI Access Control Modal */}
            <Dialog
              open={isUIAccessControlModalVisible}
              onOpenChange={(open) => !open && handleUIAccessControlCancel()}
            >
              <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[600px]">
                <DialogHeader>
                  <DialogTitle>UI Access Control Settings</DialogTitle>
                </DialogHeader>
                <UIAccessControlForm
                  accessToken={accessToken}
                  onSuccess={() => {
                    handleUIAccessControlOk();
                    toast.success("UI Access Control settings updated successfully");
                  }}
                />
              </DialogContent>
            </Dialog>
          </div>
          <Alert variant="info">
            <Info />
            <AlertTitle>Login without SSO</AlertTitle>
            <AlertDescription>
              If you need to login without sso, you can access{" "}
              <a href={nonSssoUrl} target="_blank" rel="noopener noreferrer">
                <b>{nonSssoUrl}</b>{" "}
              </a>
            </AlertDescription>
          </Alert>
        </>
      ),
    },
    {
      key: "scim",
      label: "SCIM",
      children: <SCIMConfig accessToken={accessToken} userID={userID} proxySettings={proxySettings} />,
    },
    {
      key: "ui-settings",
      label: "UI Settings",
      children: (
        <div className="flex flex-col gap-4">
          <UISettings />
          <UserBannerSettings />
        </div>
      ),
    },
    {
      key: "logging-settings",
      label: "Logging Settings",
      children: <LoggingSettings />,
    },
    {
      key: "hashicorp-vault",
      label: "Hashicorp Vault",
      children: <HashicorpVault />,
    },
    {
      key: "plugins",
      label: "Plugins",
      children: <PluginSettings />,
    },
  ];

  return (
    <div className="w-full m-2 mt-2 p-8">
      <h2 className="mb-2 text-base font-semibold text-foreground">Admin Access</h2>
      <p className="mb-4 text-sm text-foreground">Go to &apos;Internal Users&apos; page to add other admins.</p>
      <Tabs defaultValue={tabItems[0].key}>
        <TabsList variant="line" className="mb-4 h-auto flex-wrap">
          {tabItems.map((item) => (
            <TabsTrigger key={item.key} value={item.key} className="flex-none">
              {item.label}
            </TabsTrigger>
          ))}
        </TabsList>
        {tabItems.map((item) => (
          <TabsContent key={item.key} value={item.key}>
            {item.children}
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
};

export default AdminPanel;
