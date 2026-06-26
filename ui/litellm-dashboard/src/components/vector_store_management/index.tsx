import React, { useState, useEffect } from "react";
import {
  Icon,
  Button as TremorButton,
  Col,
  Text,
  Grid,
  TabGroup,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
} from "@tremor/react";
import { RefreshIcon } from "@heroicons/react/outline";
import { vectorStoreListCall, vectorStoreDeleteCall, credentialListCall, CredentialItem } from "../networking";
import { VectorStore } from "./types";
import VectorStoreTable from "./VectorStoreTable";
import VectorStoreForm from "./VectorStoreForm";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import VectorStoreInfoView from "./vector_store_info";
import CreateVectorStore from "./CreateVectorStore";
import TestVectorStoreTab from "./TestVectorStoreTab";
import { isAdminRole } from "@/utils/roles";
import NotificationsManager from "../molecules/notifications_manager";
import { useTranslation } from "react-i18next";

interface VectorStoreProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
}

const VectorStoreManagement: React.FC<VectorStoreProps> = ({ accessToken, userID, userRole }) => {
  const { t } = useTranslation();
  const [vectorStores, setVectorStores] = useState<VectorStore[]>([]);
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [vectorStoreToDelete, setVectorStoreToDelete] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState("");
  const [credentials, setCredentials] = useState<CredentialItem[]>([]);
  const [selectedVectorStoreId, setSelectedVectorStoreId] = useState<string | null>(null);
  const [editVectorStore, setEditVectorStore] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const fetchVectorStores = async () => {
    if (!accessToken) return;
    try {
      const response = await vectorStoreListCall(accessToken);
      console.log("List vector stores response:", response);
      setVectorStores(response.data || []);
    } catch (error) {
      console.error("Error fetching vector stores:", error);
      NotificationsManager.fromBackend(t("vectorStoreManagement.index.fetchVectorStoresFailed", { error }));
    }
  };

  const fetchCredentials = async () => {
    if (!accessToken) return;
    try {
      const response = await credentialListCall(accessToken);
      console.log("List credentials response:", response);
      setCredentials(response.credentials || []);
    } catch (error) {
      console.error("Error fetching credentials:", error);
      NotificationsManager.fromBackend(t("vectorStoreManagement.index.fetchCredentialsFailed", { error }));
    }
  };

  const handleRefreshClick = () => {
    fetchVectorStores();
    fetchCredentials();
    const currentDate = new Date();
    setLastRefreshed(currentDate.toLocaleString());
  };

  const handleDelete = async (vectorStoreId: string) => {
    setVectorStoreToDelete(vectorStoreId);
    setIsDeleteModalOpen(true);
  };

  const handleView = (vectorStoreId: string) => {
    setSelectedVectorStoreId(vectorStoreId);
    setEditVectorStore(false);
  };

  const handleEdit = (vectorStoreId: string) => {
    setSelectedVectorStoreId(vectorStoreId);
    setEditVectorStore(true);
  };

  const handleCloseInfo = () => {
    setSelectedVectorStoreId(null);
    setEditVectorStore(false);
    fetchVectorStores();
  };

  const confirmDelete = async () => {
    if (!accessToken || !vectorStoreToDelete) return;
    setIsDeleting(true);
    try {
      await vectorStoreDeleteCall(accessToken, vectorStoreToDelete);
      NotificationsManager.success(t("vectorStoreManagement.index.deleteSuccess"));
      fetchVectorStores();
    } catch (error) {
      console.error("Error deleting vector store:", error);
      NotificationsManager.fromBackend(t("vectorStoreManagement.index.deleteFailed", { error }));
    } finally {
      setIsDeleting(false);
      setIsDeleteModalOpen(false);
      setVectorStoreToDelete(null);
    }
  };

  const handleCreateSuccess = () => {
    setIsCreateModalVisible(false);
    fetchVectorStores();
  };

  const handleVectorStoreCreated = (vectorStoreId: string) => {
    console.log("Vector store created:", vectorStoreId);
    fetchVectorStores();
    // Optionally switch to the manage tab
  };

  useEffect(() => {
    fetchVectorStores();
    fetchCredentials();
  }, [accessToken]);

  return selectedVectorStoreId ? (
    <div className="w-full h-full">
      <VectorStoreInfoView
        vectorStoreId={selectedVectorStoreId}
        onClose={handleCloseInfo}
        accessToken={accessToken}
        is_admin={isAdminRole(userRole || "")}
        editVectorStore={editVectorStore}
      />
    </div>
  ) : (
    <div className="w-full mx-4 h-[75vh]">
      <div className="gap-2 p-8 h-[75vh] w-full mt-2">
        <div className="flex justify-between mt-2 w-full items-center mb-4">
          <h1>{t("vectorStoreManagement.index.pageTitle")}</h1>
          <div className="flex items-center space-x-2">
            {lastRefreshed && <Text>{t("vectorStoreManagement.index.lastRefreshed", { time: lastRefreshed })}</Text>}
            <Icon
              icon={RefreshIcon}
              variant="shadow"
              size="xs"
              className="self-center cursor-pointer"
              onClick={handleRefreshClick}
            />
          </div>
        </div>

        <Text className="mb-4">
          <p>{t("vectorStoreManagement.index.pageDescription")}</p>
        </Text>

        <TabGroup>
          <TabList className="mb-6">
            <Tab>{t("vectorStoreManagement.index.tabCreate")}</Tab>
            <Tab>{t("vectorStoreManagement.index.tabManage")}</Tab>
            <Tab>{t("vectorStoreManagement.index.tabTest")}</Tab>
          </TabList>

          <TabPanels>
            {/* Tab 1: Create Vector Store */}
            <TabPanel>
              <CreateVectorStore accessToken={accessToken} onSuccess={handleVectorStoreCreated} />
            </TabPanel>

            {/* Tab 2: Manage Vector Stores */}
            <TabPanel>
              <TremorButton className="mb-4" onClick={() => setIsCreateModalVisible(true)}>
                {t("vectorStoreManagement.index.addVectorStore")}
              </TremorButton>

              <Grid numItems={1} className="gap-2 pt-2 pb-2 w-full mt-2">
                <Col numColSpan={1}>
                  <VectorStoreTable
                    data={vectorStores}
                    onView={handleView}
                    onEdit={handleEdit}
                    onDelete={handleDelete}
                  />
                </Col>
              </Grid>
            </TabPanel>

            {/* Tab 3: Test Vector Store */}
            <TabPanel>
              <TestVectorStoreTab accessToken={accessToken} vectorStores={vectorStores} />
            </TabPanel>
          </TabPanels>
        </TabGroup>

        {/* Create Vector Store Modal */}
        <VectorStoreForm
          isVisible={isCreateModalVisible}
          onCancel={() => setIsCreateModalVisible(false)}
          onSuccess={handleCreateSuccess}
          accessToken={accessToken}
          credentials={credentials}
        />

        {/* Delete Confirmation Modal */}
        <DeleteResourceModal
          isOpen={isDeleteModalOpen}
          title={t("vectorStoreManagement.index.deleteModalTitle")}
          message={t("vectorStoreManagement.index.deleteModalMessage")}
          resourceInformationTitle={t("vectorStoreManagement.index.deleteModalInfoTitle")}
          resourceInformation={[
            { label: t("vectorStoreManagement.index.vectorStoreIdLabel"), value: vectorStoreToDelete, code: true },
          ]}
          onCancel={() => setIsDeleteModalOpen(false)}
          onOk={confirmDelete}
          confirmLoading={isDeleting}
        />
      </div>
    </div>
  );
};

export default VectorStoreManagement;
