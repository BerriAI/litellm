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
import React, { useEffect, useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import TableIconActionButton from "../common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import NotificationsManager from "../molecules/notifications_manager";
import { budgetDeleteCall, getBudgetList } from "../networking";
import BudgetModal from "./budget_modal";
import EditBudgetModal from "./edit_budget_modal";
import { CREATE_END_USER_CURL_COMMAND, CHAT_COMPLETIONS_CURL_COMMAND, OPENAI_SDK_PYTHON_CODE } from "./constants";

interface BudgetSettingsPageProps {
  accessToken: string | null;
}

export interface budgetItem {
  budget_id: string;
  max_budget: string | null;
  rpm_limit: number | null;
  tpm_limit: number | null;
  updated_at: string;
}

const BudgetPanel: React.FC<BudgetSettingsPageProps> = ({ accessToken }) => {
  const [isCreateModelVisible, setIsCreateModelVisible] = useState(false);
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);
  const [selectedBudget, setSelectedBudget] = useState<budgetItem | null>(null);
  const [budgetList, setBudgetList] = useState<budgetItem[]>([]);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isDeleteModalVisible, setIsDeleteModalVisible] = useState(false);
  useEffect(() => {
    if (!accessToken) {
      return;
    }
    getBudgetList(accessToken).then((data) => {
      setBudgetList(data);
    });
  }, [accessToken]);

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
    setIsDeleting(true);
    try {
      await budgetDeleteCall(accessToken, selectedBudget.budget_id);
      NotificationsManager.success("Budget deleted.");
      await handleUpdateCall();
    } catch (error) {
      console.error("Error deleting budget:", error);
      if (typeof NotificationsManager.fromBackend === "function") {
        NotificationsManager.fromBackend("Failed to delete budget");
      } else {
        NotificationsManager.info("Failed to delete budget");
      }
    } finally {
      setIsDeleting(false);
      setIsDeleteModalVisible(false);
      setSelectedBudget(null);
    }
  };

  const handleDeleteCancel = () => {
    setIsDeleteModalVisible(false);
  };

  const handleUpdateCall = async () => {
    if (accessToken == null) {
      return;
    }
    getBudgetList(accessToken).then((data) => {
      setBudgetList(data);
    });
  };

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <Button size="sm" variant="primary" className="mb-2" onClick={() => setIsCreateModelVisible(true)}>
        + Create Budget
      </Button>
      <TabGroup>
        <TabList>
          <Tab>Budgets</Tab>
          <Tab>Examples</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <div className="mt-6">
              <BudgetModal
                accessToken={accessToken}
                isModalVisible={isCreateModelVisible}
                setIsModalVisible={setIsCreateModelVisible}
                setBudgetList={setBudgetList}
              />
              {selectedBudget && (
                <EditBudgetModal
                  accessToken={accessToken}
                  isModalVisible={isEditModalVisible}
                  setIsModalVisible={setIsEditModalVisible}
                  setBudgetList={setBudgetList}
                  existingBudget={selectedBudget}
                  handleUpdateCall={handleUpdateCall}
                />
              )}
              <Card>
                <Text>Create a budget to assign to customers.</Text>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>Budget ID</TableHeaderCell>
                      <TableHeaderCell>Max Budget</TableHeaderCell>
                      <TableHeaderCell>TPM</TableHeaderCell>
                      <TableHeaderCell>RPM</TableHeaderCell>
                    </TableRow>
                  </TableHead>

                  <TableBody>
                    {budgetList
                      .slice() // Creates a shallow copy to avoid mutating the original array
                      .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()) // Sort by updated_at in descending order
                      .map((value: budgetItem, index: number) => (
                        <TableRow key={index}>
                          <TableCell>{value.budget_id}</TableCell>
                          <TableCell>{value.max_budget ? value.max_budget : "n/a"}</TableCell>
                          <TableCell>{value.tpm_limit ? value.tpm_limit : "n/a"}</TableCell>
                          <TableCell>{value.rpm_limit ? value.rpm_limit : "n/a"}</TableCell>
                          <TableIconActionButton
                            variant="Edit"
                            tooltipText="Edit budget"
                            onClick={() => handleEditCall(value)}
                            dataTestId="edit-budget-button"
                          />
                          <TableIconActionButton
                            variant="Delete"
                            tooltipText="Delete budget"
                            onClick={() => handleDeleteClick(value)}
                            dataTestId="delete-budget-button"
                          />
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </Card>
              <DeleteResourceModal
                isOpen={isDeleteModalVisible}
                title="Delete Budget?"
                message="Are you sure you want to delete this budget? This action cannot be undone."
                resourceInformationTitle="Budget Information"
                resourceInformation={[
                  { label: "Budget ID", value: selectedBudget?.budget_id, code: true },
                  { label: "Max Budget", value: selectedBudget?.max_budget },
                  { label: "TPM", value: selectedBudget?.tpm_limit },
                  { label: "RPM", value: selectedBudget?.rpm_limit },
                ]}
                onCancel={handleDeleteCancel}
                onOk={handleDeleteConfirm}
                confirmLoading={isDeleting}
              />
            </div>
          </TabPanel>
          <TabPanel>
            <div className="mt-6">
              <Text className="text-base">How to use budget id</Text>
              <TabGroup>
                <TabList>
                  <Tab>Assign Budget to Customer</Tab>
                  <Tab>Test it (Curl)</Tab>
                  <Tab>Test it (OpenAI SDK)</Tab>
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
