/**
 * The parent pane, showing list of budgets
 *
 */

import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";
import {
  Button,
  Card,
  Icon,
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
import NotificationsManager from "../molecules/notifications_manager";
import { budgetDeleteCall, getBudgetList } from "../networking";
import BudgetModal from "./budget_modal";
import EditBudgetModal from "./edit_budget_modal";

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
                  <Icon
                    icon={PencilAltIcon}
                    size="sm"
                    className="cursor-pointer"
                    onClick={() => handleEditCall(value)}
                  />
                  <Icon
                    icon={TrashIcon}
                    size="sm"
                    className="cursor-pointer hover:text-red-500"
                    onClick={() => handleDeleteClick(value)}
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
      <div className="mt-5">
        <Text className="text-base">How to use budget id</Text>
        <TabGroup>
          <TabList>
            <Tab>Assign Budget to Customer</Tab>
            <Tab>Test it (Curl)</Tab>

            <Tab>Test it (OpenAI SDK)</Tab>
          </TabList>
          <TabPanels>
            <TabPanel>
              <SyntaxHighlighter language="bash">
                {`
curl -X POST --location '<your_proxy_base_url>/end_user/new' \

-H 'Authorization: Bearer <your-master-key>' \

-H 'Content-Type: application/json' \

-d '{"user_id": "my-customer-id', "budget_id": "<BUDGET_ID>"}' # 👈 KEY CHANGE

            `}
              </SyntaxHighlighter>
            </TabPanel>
            <TabPanel>
              <SyntaxHighlighter language="bash">
                {`
curl -X POST --location '<your_proxy_base_url>/chat/completions' \

-H 'Authorization: Bearer <your-master-key>' \

-H 'Content-Type: application/json' \

-d '{
  "model": "gpt-3.5-turbo', 
  "messages":[{"role": "user", "content": "Hey, how's it going?"}],
  "user": "my-customer-id"
}' # 👈 KEY CHANGE

            `}
              </SyntaxHighlighter>
            </TabPanel>
            <TabPanel>
              <SyntaxHighlighter language="python">
                {`from openai import OpenAI
client = OpenAI(
  base_url="<your_proxy_base_url>",
  api_key="<your_proxy_key>"
)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  user="my-customer-id"
)

print(completion.choices[0].message)`}
              </SyntaxHighlighter>
            </TabPanel>
          </TabPanels>
        </TabGroup>
      </div>
    </div>
  );
};

export default BudgetPanel;
