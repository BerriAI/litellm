/**
 * Allow proxy admin to add other people to view global spend
 * Use this to avoid sharing master key with others
 */
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  Button,
  Callout,
  Card,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "@tremor/react";
import { Alert, Button as Button2, Form, Input, Modal, Space, Tabs, Typography } from "antd";
import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import NewBadge from "./common_components/NewBadge";
import { useBaseUrl } from "./constants";
import NotificationsManager from "./molecules/notifications_manager";
import { addAllowedIP, deleteAllowedIP, getAllowedIPs, getSSOSettings } from "./networking";
import SCIMConfig from "./SCIM";
import LoggingSettings from "./Settings/AdminSettings/LoggingSettings/LoggingSettings";
import SSOSettings from "./Settings/AdminSettings/SSOSettings/SSOSettings";
import UISettings from "./Settings/AdminSettings/UISettings/UISettings";
import HashicorpVault from "./Settings/AdminSettings/HashicorpVault/HashicorpVault";
import SSOModals from "./SSOModals";
import UIAccessControlForm from "./UIAccessControlForm";

const { Title, Paragraph, Text } = Typography;

interface AdminPanelProps {
  proxySettings?: any;
}

const AdminPanel: React.FC<AdminPanelProps> = ({ proxySettings }) => {
  const { t } = useTranslation();
  const { premiumUser, accessToken, userId: userID } = useAuthorized();
  const [form] = Form.useForm();
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
  const all_ip_address_allowed = t("adminPanel.allIPAddressesAllowed");

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
        NotificationsManager.fromBackend(t("adminPanel.notifications.premiumOnlyFeature"));
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
      NotificationsManager.fromBackend(t("adminPanel.fetchAllowedIpsFailed", { error }));
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
        NotificationsManager.success(t("adminPanel.notifications.ipAddedSuccess"));
      }
    } catch (error) {
      console.error("Error adding IP:", error);
      NotificationsManager.fromBackend(t("adminPanel.addIpFailed", { error }));
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
        NotificationsManager.success(t("adminPanel.notifications.ipDeletedSuccess"));
      } catch (error) {
        console.error("Error deleting IP:", error);
        NotificationsManager.fromBackend(t("adminPanel.deleteIpFailed", { error }));
      } finally {
        setIsDeleteIPModalVisible(false);
        setIPToDelete(null);
      }
    }
  };

  const handleAddSSOOk = () => {
    setIsAddSSOModalVisible(false);
    form.resetFields();
    if (accessToken && premiumUser) {
      checkSSOConfiguration();
    }
  };

  const handleAddSSOCancel = () => {
    setIsAddSSOModalVisible(false);
    form.resetFields();
  };

  const handleShowInstructions = (formValues: Record<string, any>) => {
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
      label: t("adminPanel.tabs.ssoSettings"),
      children: <SSOSettings />,
    },
    {
      key: "security-settings",
      label: t("adminPanel.tabs.securitySettings"),
      children: (
        <>
          <Card>
            <Title level={4}> ✨ {t("adminPanel.securityCard.title")}</Title>
            <Alert
              message={t("adminPanel.securityCard.deprecatedAlertMessage")}
              description={t("adminPanel.securityCard.deprecatedAlertDescription")}
              type="warning"
              showIcon
            />
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
                  {ssoConfigured ? t("adminPanel.securityCard.editSSOSettings") : t("adminPanel.securityCard.addSSO")}
                </Button>
              </div>
              <div>
                <Button style={{ width: "150px" }} onClick={handleShowAllowedIPs}>
                  {t("adminPanel.securityCard.allowedIPs")}
                </Button>
              </div>
              <div>
                <Button
                  style={{ width: "150px" }}
                  onClick={() =>
                    premiumUser === true
                      ? setIsUIAccessControlModalVisible(true)
                      : NotificationsManager.fromBackend(t("adminPanel.notifications.premiumOnlyUIAccessControl"))
                  }
                >
                  {t("adminPanel.securityCard.uiAccessControl")}
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
            <Modal
              title={t("adminPanel.manageAllowedIPModal.title")}
              width={800}
              open={isAllowedIPModalVisible}
              onCancel={() => setIsAllowedIPModalVisible(false)}
              footer={[
                <Button className="mx-1" key="add" onClick={() => setIsAddIPModalVisible(true)}>
                  {t("adminPanel.manageAllowedIPModal.addIPAddress")}
                </Button>,
                <Button key="close" onClick={() => setIsAllowedIPModalVisible(false)}>
                  {t("common.close")}
                </Button>,
              ]}
            >
              <Table>
                <TableHead>
                  <TableRow>
                    <TableHeaderCell>{t("adminPanel.manageAllowedIPModal.ipAddressColumn")}</TableHeaderCell>
                    <TableHeaderCell className="text-right">
                      {t("adminPanel.manageAllowedIPModal.actionColumn")}
                    </TableHeaderCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {allowedIPs.map((ip, index) => (
                    <TableRow key={index}>
                      <TableCell>{ip}</TableCell>
                      <TableCell className="text-right">
                        {ip !== all_ip_address_allowed && (
                          <Button onClick={() => handleDeleteIP(ip)} color="red" size="xs">
                            {t("common.delete")}
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Modal>

            <Modal
              title={t("adminPanel.addIPModal.title")}
              open={isAddIPModalVisible}
              onCancel={() => setIsAddIPModalVisible(false)}
              footer={null}
            >
              <Form onFinish={handleAddIP}>
                <Form.Item name="ip" rules={[{ required: true, message: t("adminPanel.addIPModal.inputRequired") }]}>
                  <Input placeholder={t("adminPanel.addIPModal.inputPlaceholder")} />
                </Form.Item>
                <Form.Item>
                  <Button2 htmlType="submit">{t("adminPanel.addIPModal.submitButton")}</Button2>
                </Form.Item>
              </Form>
            </Modal>

            <Modal
              title={t("adminPanel.deleteIPModal.title")}
              open={isDeleteIPModalVisible}
              onCancel={() => setIsDeleteIPModalVisible(false)}
              onOk={confirmDeleteIP}
              footer={[
                <Button className="mx-1" key="delete" onClick={() => confirmDeleteIP()}>
                  {t("common.yes")}
                </Button>,
                <Button key="close" onClick={() => setIsDeleteIPModalVisible(false)}>
                  {t("common.close")}
                </Button>,
              ]}
            >
              <Text>{t("adminPanel.deleteIPModal.confirmText", { ip: ipToDelete })}</Text>
            </Modal>

            {/* UI Access Control Modal */}
            <Modal
              title={t("adminPanel.uiAccessControlModal.title")}
              open={isUIAccessControlModalVisible}
              width={600}
              footer={null}
              onOk={handleUIAccessControlOk}
              onCancel={handleUIAccessControlCancel}
            >
              <UIAccessControlForm
                accessToken={accessToken}
                onSuccess={() => {
                  handleUIAccessControlOk();
                  NotificationsManager.success(t("adminPanel.uiAccessControlModal.updateSuccess"));
                }}
              />
            </Modal>
          </div>
          <Callout title={t("adminPanel.loginWithoutSSO.title")} color="teal">
            {t("adminPanel.loginWithoutSSO.description")}{" "}
            <a href={nonSssoUrl} target="_blank" rel="noopener noreferrer">
              <b>{nonSssoUrl}</b>{" "}
            </a>
          </Callout>
        </>
      ),
    },
    {
      key: "scim",
      label: t("adminPanel.tabs.scim"),
      children: <SCIMConfig accessToken={accessToken} userID={userID} proxySettings={proxySettings} />,
    },
    {
      key: "ui-settings",
      label: (
        <Space>
          <Text>
            {t("adminPanel.tabs.uiSettings")} <NewBadge />
          </Text>
        </Space>
      ),
      children: <UISettings />,
    },
    {
      key: "logging-settings",
      label: t("adminPanel.tabs.loggingSettings"),
      children: <LoggingSettings />,
    },
    {
      key: "hashicorp-vault",
      label: t("adminPanel.tabs.hashicorpVault"),
      children: <HashicorpVault />,
    },
  ];

  return (
    <div className="w-full m-2 mt-2 p-8">
      <Title level={4}>{t("adminPanel.title")} </Title>
      <Paragraph>{t("adminPanel.goToInternalUsers")}</Paragraph>
      <Tabs items={tabItems} />
    </div>
  );
};

export default AdminPanel;
