/**
 * The parent pane, showing list of budgets.
 * Migrated to shadcn in phase 1. See docs/BLUEPRINT.md.
 */

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import React, { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import TableIconActionButton from "../common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import NotificationsManager from "../molecules/notifications_manager";
import { useBudgets, useDeleteBudget } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import BudgetModal from "./budget_modal";
import EditBudgetModal from "./edit_budget_modal";
import {
  CREATE_END_USER_CURL_COMMAND,
  CHAT_COMPLETIONS_CURL_COMMAND,
  OPENAI_SDK_PYTHON_CODE,
} from "./constants";

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
  const [isCreateModelVisible, setIsCreateModelVisible] = useState(false);
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);
  const [selectedBudget, setSelectedBudget] = useState<budgetItem | null>(null);
  const [isDeleteModalVisible, setIsDeleteModalVisible] = useState(false);

  const { data: budgetList = [] } = useBudgets();
  const deleteBudget = useDeleteBudget();

  const handleEditCall = (budget: budgetItem) => {
    if (accessToken == null) return;
    setSelectedBudget(budget);
    setIsEditModalVisible(true);
  };

  const handleDeleteClick = (budget: budgetItem) => {
    setSelectedBudget(budget);
    setIsDeleteModalVisible(true);
  };

  const handleDeleteConfirm = async () => {
    if (!selectedBudget || accessToken == null) return;
    try {
      await deleteBudget.mutateAsync(selectedBudget.budget_id);
      NotificationsManager.success("Budget deleted.");
    } catch (error) {
      console.error("Error deleting budget:", error);
      if (typeof NotificationsManager.fromBackend === "function") {
        NotificationsManager.fromBackend("Failed to delete budget");
      } else {
        NotificationsManager.info("Failed to delete budget");
      }
    } finally {
      setIsDeleteModalVisible(false);
      setSelectedBudget(null);
    }
  };

  const handleDeleteCancel = () => setIsDeleteModalVisible(false);

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <Button size="sm" className="mb-2" onClick={() => setIsCreateModelVisible(true)}>
        + Create Budget
      </Button>
      <Tabs defaultValue="budgets">
        <TabsList>
          <TabsTrigger value="budgets">Budgets</TabsTrigger>
          <TabsTrigger value="examples">Examples</TabsTrigger>
        </TabsList>
        <TabsContent value="budgets">
          <div className="mt-6">
            <BudgetModal
              isModalVisible={isCreateModelVisible}
              setIsModalVisible={setIsCreateModelVisible}
            />
            {selectedBudget && (
              <EditBudgetModal
                isModalVisible={isEditModalVisible}
                setIsModalVisible={setIsEditModalVisible}
                existingBudget={selectedBudget}
              />
            )}
            <Card className="p-6">
              <p className="text-sm text-muted-foreground mb-3">
                Create a budget to assign to customers.
              </p>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Budget ID</TableHead>
                    <TableHead>Max Budget</TableHead>
                    <TableHead>TPM</TableHead>
                    <TableHead>RPM</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>

                <TableBody>
                  {budgetList
                    .slice()
                    .sort(
                      (a, b) =>
                        new Date(b.updated_at).getTime() -
                        new Date(a.updated_at).getTime(),
                    )
                    .map((value: budgetItem) => (
                      <TableRow key={value.budget_id}>
                        <TableCell>{value.budget_id}</TableCell>
                        <TableCell>
                          {value.max_budget ? value.max_budget : "n/a"}
                        </TableCell>
                        <TableCell>
                          {value.tpm_limit ? value.tpm_limit : "n/a"}
                        </TableCell>
                        <TableCell>
                          {value.rpm_limit ? value.rpm_limit : "n/a"}
                        </TableCell>
                        <TableCell>
                          <div className="flex gap-1">
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
                          </div>
                        </TableCell>
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
              confirmLoading={deleteBudget.isPending}
            />
          </div>
        </TabsContent>
        <TabsContent value="examples">
          <div className="mt-6">
            <p className="text-base font-medium">How to use budget id</p>
            <Tabs defaultValue="assign">
              <TabsList>
                <TabsTrigger value="assign">Assign Budget to Customer</TabsTrigger>
                <TabsTrigger value="curl">Test it (Curl)</TabsTrigger>
                <TabsTrigger value="openai">Test it (OpenAI SDK)</TabsTrigger>
              </TabsList>
              <TabsContent value="assign">
                <SyntaxHighlighter language="bash">
                  {CREATE_END_USER_CURL_COMMAND}
                </SyntaxHighlighter>
              </TabsContent>
              <TabsContent value="curl">
                <SyntaxHighlighter language="bash">
                  {CHAT_COMPLETIONS_CURL_COMMAND}
                </SyntaxHighlighter>
              </TabsContent>
              <TabsContent value="openai">
                <SyntaxHighlighter language="python">
                  {OPENAI_SDK_PYTHON_CODE}
                </SyntaxHighlighter>
              </TabsContent>
            </Tabs>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default BudgetPanel;
