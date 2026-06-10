/**
 * The parent pane, showing list of budgets
 *
 */

import {
  Button,
  Card,
  Tab,
  TabGroup,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  TabList,
  TabPanel,
  TabPanels,
  Text,
} from "@tremor/react";
import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import TableIconActionButton from "../common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import NotificationsManager from "../molecules/notifications_manager";
import { useBudgets, useDeleteBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import BudgetModal from "./budget_modal";
import EditBudgetModal from "./edit_budget_modal";
import { CREATE_END_USER_CURL_COMMAND, CHAT_COMPLETIONS_CURL_COMMAND, OPENAI_SDK_PYTHON_CODE } from "./constants";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";

interface BudgetSettingsPageProps {
  accessToken: string | null;
}

export interface budgetItem {
  budget_id: string;
  max_budget: number | null;
  rpm_limit: number | null;
  tpm_limit: number | null;
  updated_at: string;
}

const BudgetPanel: React.FC<BudgetSettingsPageProps> = ({ accessToken }) => {
  const { t } = useTranslation();
  const [isCreateModelVisible, setIsCreateModelVisible] = useState(false);
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);
  const [selectedBudget, setSelectedBudget] = useState<budgetItem | null>(null);
  const [isDeleteModalVisible, setIsDeleteModalVisible] = useState(false);

  const { userRole } = useAuthorized();
  // Admin Viewer follows the read-parity rule: see budgets, no writes.
  const canModify = isProxyAdminRole(userRole ?? "");

  const { data: budgetList = [] } = useBudgets();
  const deleteBudget = useDeleteBudget();

  const handleEditCall = async (budget: budgetItem) => {
    if (accessToken == null) {
      return;
    }
    setSelectedBudget(budget);
    setIsEditModalVisible(true);
  };

  const handleDeleteClick = (budget: budgetItem) => {
    setSelectedBudget(budget);
    setIsDeleteModalVisible(true);
  };

  const handleDeleteConfirm = async () => {
    if (!selectedBudget || accessToken == null) {
      return;
    }
    try {
      await deleteBudget.mutateAsync(selectedBudget.budget_id);
      NotificationsManager.success(t("budgets.budgetPanel.budgetDeleted"));
    } catch (error) {
      console.error("Error deleting budget:", error);
      if (typeof NotificationsManager.fromBackend === "function") {
        NotificationsManager.fromBackend(t("budgets.budgetPanel.failedToDelete"));
      } else {
        NotificationsManager.info(t("budgets.budgetPanel.failedToDelete"));
      }
    } finally {
      setIsDeleteModalVisible(false);
      setSelectedBudget(null);
    }
  };

  const handleDeleteCancel = () => {
    setIsDeleteModalVisible(false);
  };

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      {canModify && (
        <Button size="sm" variant="primary" className="mb-2" onClick={() => setIsCreateModelVisible(true)}>
          {t("budgets.budgetPanel.createBudgetButton")}
        </Button>
      )}
      <TabGroup>
        <TabList>
          <Tab>{t("budgets.budgetPanel.tabBudgets")}</Tab>
          <Tab>{t("budgets.budgetPanel.tabExamples")}</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <div className="mt-6">
              <BudgetModal isModalVisible={isCreateModelVisible} setIsModalVisible={setIsCreateModelVisible} />
              {selectedBudget && (
                <EditBudgetModal
                  isModalVisible={isEditModalVisible}
                  setIsModalVisible={setIsEditModalVisible}
                  existingBudget={selectedBudget}
                />
              )}
              <Card>
                <Text>{t("budgets.budgetPanel.createBudgetDescription")}</Text>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>{t("budgets.budgetPanel.colBudgetId")}</TableHeaderCell>
                      <TableHeaderCell>{t("budgets.budgetPanel.colMaxBudget")}</TableHeaderCell>
                      <TableHeaderCell>{t("budgets.budgetPanel.colTpm")}</TableHeaderCell>
                      <TableHeaderCell>{t("budgets.budgetPanel.colRpm")}</TableHeaderCell>
                    </TableRow>
                  </TableHead>

                  <TableBody>
                    {budgetList
                      .slice()
                      .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
                      .map((value: budgetItem) => (
                        <TableRow key={value.budget_id}>
                          <TableCell>{value.budget_id}</TableCell>
                          <TableCell>{value.max_budget ? value.max_budget : "n/a"}</TableCell>
                          <TableCell>{value.tpm_limit ? value.tpm_limit : "n/a"}</TableCell>
                          <TableCell>{value.rpm_limit ? value.rpm_limit : "n/a"}</TableCell>
                          {canModify && (
                            <>
                              <TableIconActionButton
                                variant="Edit"
                                tooltipText={t("budgets.budgetPanel.editBudgetTooltip")}
                                onClick={() => handleEditCall(value)}
                                dataTestId="edit-budget-button"
                              />
                              <TableIconActionButton
                                variant="Delete"
                                tooltipText={t("budgets.budgetPanel.deleteBudgetTooltip")}
                                onClick={() => handleDeleteClick(value)}
                                dataTestId="delete-budget-button"
                              />
                            </>
                          )}
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </Card>
              <DeleteResourceModal
                isOpen={isDeleteModalVisible}
                title={t("budgets.budgetPanel.deleteTitle")}
                message={t("budgets.budgetPanel.deleteMessage")}
                resourceInformationTitle={t("budgets.budgetPanel.budgetInformation")}
                resourceInformation={[
                  { label: t("budgets.budgetPanel.colBudgetId"), value: selectedBudget?.budget_id, code: true },
                  { label: t("budgets.budgetPanel.colMaxBudget"), value: selectedBudget?.max_budget },
                  { label: t("budgets.budgetPanel.colTpm"), value: selectedBudget?.tpm_limit },
                  { label: t("budgets.budgetPanel.colRpm"), value: selectedBudget?.rpm_limit },
                ]}
                onCancel={handleDeleteCancel}
                onOk={handleDeleteConfirm}
                confirmLoading={deleteBudget.isPending}
              />
            </div>
          </TabPanel>
          <TabPanel>
            <div className="mt-6">
              <Text className="text-base">{t("budgets.budgetPanel.howToUseBudgetId")}</Text>
              <TabGroup>
                <TabList>
                  <Tab>{t("budgets.budgetPanel.tabAssignBudget")}</Tab>
                  <Tab>{t("budgets.budgetPanel.tabTestCurl")}</Tab>
                  <Tab>{t("budgets.budgetPanel.tabTestSdk")}</Tab>
                </TabList>
                <TabPanels>
                  <TabPanel>
                    <SyntaxHighlighter language="bash">{CREATE_END_USER_CURL_COMMAND}</SyntaxHighlighter>
                  </TabPanel>
                  <TabPanel>
                    <SyntaxHighlighter language="bash">{CHAT_COMPLETIONS_CURL_COMMAND}</SyntaxHighlighter>
                  </TabPanel>
                  <TabPanel>
                    <SyntaxHighlighter language="python">{OPENAI_SDK_PYTHON_CODE}</SyntaxHighlighter>
                  </TabPanel>
                </TabPanels>
              </TabGroup>
            </div>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default BudgetPanel;
