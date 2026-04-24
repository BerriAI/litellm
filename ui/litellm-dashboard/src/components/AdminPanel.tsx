/**
 * Allow proxy admin to add other people to view global spend
 * Use this to avoid sharing master key with others
 */
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import React, { useEffect, useRef, useState } from "react";
import { AlertTriangle, LogIn } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import NewBadge from "./common_components/NewBadge";
import { useBaseUrl } from "./constants";
import NotificationsManager from "./molecules/notifications_manager";
import { addAllowedIP, deleteAllowedIP, getAllowedIPs, getSSOSettings } from "./networking";
import SCIMConfig from "./SCIM";
import SSOSettings from "./Settings/AdminSettings/SSOSettings/SSOSettings";
import UISettings from "./Settings/AdminSettings/UISettings/UISettings";
import HashicorpVault from "./Settings/AdminSettings/HashicorpVault/HashicorpVault";
import SSOModals from "./SSOModals";
import UIAccessControlForm from "./UIAccessControlForm";

interface AdminPanelProps {
  proxySettings?: any;
}

const AdminPanel: React.FC<AdminPanelProps> = ({ proxySettings }) => {
  const { premiumUser, accessToken, userId: userID } = useAuthorized();
  // Legacy antd `Form` handle — retained as a no-op bridge so the
  // SSOModals component (which still accepts a `form` prop for back-compat)
  // has something to call `resetFields` on.
  const ssoFormHandle = useRef({
    resetFields: () => {},
    setFieldsValue: () => {},
    getFieldsValue: () => ({}),
  });
  const [isAddSSOModalVisible, setIsAddSSOModalVisible] = useState(false);
  const [isInstructionsModalVisible, setIsInstructionsModalVisible] = useState(false);
  const [isAllowedIPModalVisible, setIsAllowedIPModalVisible] = useState(false);
  const [isAddIPModalVisible, setIsAddIPModalVisible] = useState(false);
  const [isDeleteIPModalVisible, setIsDeleteIPModalVisible] = useState(false);
  const [isUIAccessControlModalVisible, setIsUIAccessControlModalVisible] = useState(false);
  const [allowedIPs, setAllowedIPs] = useState<string[]>([]);
  const [ipToDelete, setIPToDelete] = useState<string | null>(null);
  const [ssoConfigured, setSsoConfigured] = useState<boolean>(false);
  const [newIP, setNewIP] = useState<string>("");

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
        NotificationsManager.fromBackend(
          "This feature is only available for premium users. Please upgrade your account.",
        );
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
      NotificationsManager.fromBackend(`Failed to fetch allowed IPs ${error}`);
      setAllowedIPs([all_ip_address_allowed]);
    } finally {
      if (premiumUser === true) {
        setIsAllowedIPModalVisible(true);
      }
    }
  };

  const handleAddIP = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!newIP.trim()) {
      NotificationsManager.fromBackend("Please enter an IP address");
      return;
    }
    try {
      if (accessToken) {
        await addAllowedIP(accessToken, newIP.trim());
        const updatedIPs = await getAllowedIPs(accessToken);
        setAllowedIPs(updatedIPs);
        NotificationsManager.success("IP address added successfully");
        setNewIP("");
      }
    } catch (error) {
      console.error("Error adding IP:", error);
      NotificationsManager.fromBackend(`Failed to add IP address ${error}`);
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
        const updatedIPs = await getAllowedIPs(accessToken);
        setAllowedIPs(updatedIPs.length > 0 ? updatedIPs : [all_ip_address_allowed]);
        NotificationsManager.success("IP address deleted successfully");
      } catch (error) {
        console.error("Error deleting IP:", error);
        NotificationsManager.fromBackend(`Failed to delete IP address ${error}`);
      } finally {
        setIsDeleteIPModalVisible(false);
        setIPToDelete(null);
      }
    }
  };

  const handleAddSSOOk = () => {
    setIsAddSSOModalVisible(false);
    ssoFormHandle.current.resetFields();
    if (accessToken && premiumUser) {
      checkSSOConfiguration();
    }
  };

  const handleAddSSOCancel = () => {
    setIsAddSSOModalVisible(false);
    ssoFormHandle.current.resetFields();
  };

  const handleShowInstructions = (_formValues: Record<string, any>) => {
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken, premiumUser]);

  const handleUIAccessControlOk = () => {
    setIsUIAccessControlModalVisible(false);
  };

  const handleUIAccessControlCancel = () => {
    setIsUIAccessControlModalVisible(false);
  };

  return (
    <div className="w-full m-2 mt-2 p-8">
      <h1 className="text-xl font-semibold">Admin Access </h1>
      <p className="text-sm text-muted-foreground mb-4">
        Go to &apos;Internal Users&apos; page to add other admins.
      </p>
      <Tabs defaultValue="sso-settings">
        <TabsList>
          <TabsTrigger value="sso-settings">SSO Settings</TabsTrigger>
          <TabsTrigger value="security-settings">Security Settings</TabsTrigger>
          <TabsTrigger value="scim">SCIM</TabsTrigger>
          <TabsTrigger value="ui-settings">
            <span className="inline-flex items-center gap-2">
              UI Settings
              <NewBadge />
            </span>
          </TabsTrigger>
          <TabsTrigger value="hashicorp-vault">Hashicorp Vault</TabsTrigger>
        </TabsList>

        <TabsContent value="sso-settings">
          <SSOSettings />
        </TabsContent>

        <TabsContent value="security-settings">
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-3">✨ Security Settings</h3>
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>SSO Configuration Deprecated</AlertTitle>
              <AlertDescription>
                Editing SSO Settings on this page is deprecated and will be removed in a future
                version. Please use the SSO Settings tab for SSO configuration.
              </AlertDescription>
            </Alert>
            <div className="flex flex-col gap-4 mt-4 ml-2">
              <div>
                <Button className="w-[150px]" onClick={() => setIsAddSSOModalVisible(true)}>
                  {ssoConfigured ? "Edit SSO Settings" : "Add SSO"}
                </Button>
              </div>
              <div>
                <Button className="w-[150px]" onClick={handleShowAllowedIPs}>
                  Allowed IPs
                </Button>
              </div>
              <div>
                <Button
                  className="w-[150px]"
                  onClick={() =>
                    premiumUser === true
                      ? setIsUIAccessControlModalVisible(true)
                      : NotificationsManager.fromBackend("Only premium users can configure UI access control")
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
              form={ssoFormHandle.current}
              accessToken={accessToken}
              ssoConfigured={ssoConfigured}
            />

            <Dialog
              open={isAllowedIPModalVisible}
              onOpenChange={(o) => {
                if (!o) setIsAllowedIPModalVisible(false);
              }}
            >
              <DialogContent className="max-w-[800px]">
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
                            <Button
                              onClick={() => handleDeleteIP(ip)}
                              variant="destructive"
                              size="sm"
                            >
                              Delete
                            </Button>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <DialogFooter>
                  <Button onClick={() => setIsAddIPModalVisible(true)}>Add IP Address</Button>
                  <Button variant="outline" onClick={() => setIsAllowedIPModalVisible(false)}>
                    Close
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Dialog
              open={isAddIPModalVisible}
              onOpenChange={(o) => {
                if (!o) setIsAddIPModalVisible(false);
              }}
            >
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add Allowed IP Address</DialogTitle>
                </DialogHeader>
                <form onSubmit={handleAddIP} className="space-y-4">
                  <Input
                    placeholder="Enter IP address"
                    value={newIP}
                    onChange={(e) => setNewIP(e.target.value)}
                  />
                  <DialogFooter>
                    <Button type="submit">Add IP Address</Button>
                  </DialogFooter>
                </form>
              </DialogContent>
            </Dialog>

            <Dialog
              open={isDeleteIPModalVisible}
              onOpenChange={(o) => {
                if (!o) setIsDeleteIPModalVisible(false);
              }}
            >
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Confirm Delete</DialogTitle>
                </DialogHeader>
                <p>Are you sure you want to delete the IP address: {ipToDelete}?</p>
                <DialogFooter>
                  <Button variant="destructive" onClick={() => confirmDeleteIP()}>
                    Yes
                  </Button>
                  <Button variant="outline" onClick={() => setIsDeleteIPModalVisible(false)}>
                    Close
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            {/* UI Access Control Modal */}
            <Dialog
              open={isUIAccessControlModalVisible}
              onOpenChange={(o) => {
                if (!o) handleUIAccessControlCancel();
              }}
            >
              <DialogContent className="max-w-[600px]">
                <DialogHeader>
                  <DialogTitle>UI Access Control Settings</DialogTitle>
                </DialogHeader>
                <UIAccessControlForm
                  accessToken={accessToken}
                  onSuccess={() => {
                    handleUIAccessControlOk();
                    NotificationsManager.success("UI Access Control settings updated successfully");
                  }}
                />
              </DialogContent>
            </Dialog>
          </div>

          {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
          <div className="rounded-md border border-teal-200 bg-teal-50 text-teal-900 dark:bg-teal-950/30 dark:border-teal-900 dark:text-teal-200 p-4 flex gap-3">
            <LogIn className="h-5 w-5 mt-0.5 shrink-0" />
            <div>
              <p className="font-medium">Login without SSO</p>
              <p className="text-sm">
                If you need to login without sso, you can access{" "}
                <a href={nonSssoUrl} target="_blank" rel="noopener noreferrer">
                  <b>{nonSssoUrl}</b>
                </a>
              </p>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="scim">
          <SCIMConfig accessToken={accessToken} userID={userID} proxySettings={proxySettings} />
        </TabsContent>

        <TabsContent value="ui-settings">
          <UISettings />
        </TabsContent>

        <TabsContent value="hashicorp-vault">
          <HashicorpVault />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AdminPanel;
