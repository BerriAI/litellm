/**
 * The parent pane, showing list of budgets
 *
 */

import { Plus, Wallet } from "lucide-react";
import React, { useCallback, useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { PageHeader } from "@/components/shared/PageHeader";
import { ToolbarSeparator } from "@/components/shared/ToolbarSeparator";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { useBudgetList, useDeleteBudget, budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import BudgetModal from "./budget_modal";
import BudgetTable from "./BudgetTable";
import EditBudgetModal from "./edit_budget_modal";
import { CREATE_END_USER_CURL_COMMAND, CHAT_COMPLETIONS_CURL_COMMAND, OPENAI_SDK_PYTHON_CODE } from "./constants";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import { useTranslation } from "react-i18next";

interface BudgetSettingsPageProps {
  accessToken: string | null;
}

const BudgetPanel: React.FC<BudgetSettingsPageProps> = ({ accessToken }) => {
  const { t } = useTranslation("gateway");
  const [isCreateModelVisible, setIsCreateModelVisible] = useState(false);
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);
  const [selectedBudget, setSelectedBudget] = useState<budgetItem | null>(null);
  const [isDeleteModalVisible, setIsDeleteModalVisible] = useState(false);

  const { userRole } = useAuthorized();
  // Admin Viewer follows the read-parity rule: see budgets, no writes.
  const canModify = isProxyAdminRole(userRole ?? "");

  const budgetList = useBudgetList();
  const deleteBudget = useDeleteBudget();

  // Stable identities keep the memoized column defs stable; new ones remount every header and cell.
  const handleEditCall = useCallback(
    (budget: budgetItem) => {
      if (accessToken == null) {
        return;
      }
      setSelectedBudget(budget);
      setIsEditModalVisible(true);
    },
    [accessToken],
  );

  const handleDeleteClick = useCallback((budget: budgetItem) => {
    setSelectedBudget(budget);
    setIsDeleteModalVisible(true);
  }, []);

  const handleDeleteConfirm = async () => {
    if (!selectedBudget || accessToken == null) {
      return;
    }
    try {
      await deleteBudget.mutateAsync(selectedBudget.budget_id);
      NotificationsManager.success(t("budgets.notifications.deleted"));
    } catch (error) {
      console.error("Error deleting budget:", error);
      if (typeof NotificationsManager.fromBackend === "function") {
        NotificationsManager.fromBackend(t("budgets.notifications.deleteError"));
      } else {
        NotificationsManager.info(t("budgets.notifications.deleteError"));
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
    <div className="flex h-full flex-col gap-4 p-6 px-12">
      <PageHeader icon={<Wallet className="size-5" />} title={t("budgets.title")} subtitle={t("budgets.subtitle")} />
      <Tabs defaultValue="budgets" className="min-h-0 flex-1 gap-0">
        <div className="flex items-center gap-4 border-b border-border">
          {canModify && (
            <>
              <Button onClick={() => setIsCreateModelVisible(true)}>
                <Plus className="size-4" />
                {t("budgets.create")}
              </Button>
              <ToolbarSeparator className="h-6" />
            </>
          )}
          <TabsList variant="line">
            <TabsTrigger value="budgets" className="flex-none px-4">
              {t("budgets.tabs.budgets")}
            </TabsTrigger>
            <TabsTrigger value="examples" className="flex-none px-4">
              {t("budgets.tabs.examples")}
            </TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="budgets" className="flex min-h-0 flex-1 flex-col">
          <div className="flex min-h-0 flex-1 flex-col pt-6">
            <BudgetModal isModalVisible={isCreateModelVisible} setIsModalVisible={setIsCreateModelVisible} />
            {selectedBudget && (
              <EditBudgetModal
                isModalVisible={isEditModalVisible}
                setIsModalVisible={setIsEditModalVisible}
                existingBudget={selectedBudget}
              />
            )}
            <BudgetTable
              list={budgetList}
              canModify={canModify}
              onEditClick={handleEditCall}
              onDeleteClick={handleDeleteClick}
            />
            <DeleteResourceModal
              isOpen={isDeleteModalVisible}
              title={t("budgets.delete.title")}
              message={t("budgets.delete.message")}
              resourceInformationTitle={t("budgets.delete.information")}
              resourceInformation={[
                { label: t("budgets.fields.budgetId"), value: selectedBudget?.budget_id, code: true },
                { label: t("budgets.fields.maxBudget"), value: selectedBudget?.max_budget },
                { label: t("budgets.fields.tpm"), value: selectedBudget?.tpm_limit },
                { label: t("budgets.fields.rpm"), value: selectedBudget?.rpm_limit },
              ]}
              onCancel={handleDeleteCancel}
              onOk={handleDeleteConfirm}
              confirmLoading={deleteBudget.isPending}
            />
          </div>
        </TabsContent>
        <TabsContent value="examples" className="min-h-0 flex-1 overflow-y-auto">
          <div className="pt-6">
            <p className="text-base text-muted-foreground">{t("budgets.examples.description")}</p>
            <Tabs defaultValue="assign-budget">
              <TabsList variant="line" className="h-auto w-full justify-start rounded-none border-b p-0">
                <TabsTrigger value="assign-budget" className="flex-none rounded-none px-4 py-2">
                  {t("budgets.examples.assign")}
                </TabsTrigger>
                <TabsTrigger value="curl" className="flex-none rounded-none px-4 py-2">
                  {t("budgets.examples.curl")}
                </TabsTrigger>
                <TabsTrigger value="openai-sdk" className="flex-none rounded-none px-4 py-2">
                  {t("budgets.examples.sdk")}
                </TabsTrigger>
              </TabsList>
              <TabsContent value="assign-budget">
                <SyntaxHighlighter language="bash">{CREATE_END_USER_CURL_COMMAND}</SyntaxHighlighter>
              </TabsContent>
              <TabsContent value="curl">
                <SyntaxHighlighter language="bash">{CHAT_COMPLETIONS_CURL_COMMAND}</SyntaxHighlighter>
              </TabsContent>
              <TabsContent value="openai-sdk">
                <SyntaxHighlighter language="python">{OPENAI_SDK_PYTHON_CODE}</SyntaxHighlighter>
              </TabsContent>
            </Tabs>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default BudgetPanel;
