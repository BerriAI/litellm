import React, { useState } from "react";
import { Button, Card, TabPanel, Text } from "@tremor/react";
import { RefreshCw } from "lucide-react";
import CustomersTable from "@/app/(dashboard)/customers/components/CustomersTable";
import CustomerInfo from "@/app/(dashboard)/customers/components/CustomerInfo";
import CreateCustomerModal from "@/app/(dashboard)/customers/components/modals/CreateCustomerModal";
import DeleteCustomerModal from "@/app/(dashboard)/customers/components/modals/DeleteCustomerModal";
import CustomersHeaderTabs from "@/app/(dashboard)/customers/components/CustomersHeaderTabs";
import CustomersFilters from "@/app/(dashboard)/customers/components/CustomersFilters";
import type { Customer, NewCustomerData } from "@/app/(dashboard)/customers/types";
import { customerDeleteCall } from "@/components/networking";

interface CustomersViewProps {
  customers: Customer[];
  setCustomers: React.Dispatch<React.SetStateAction<Customer[]>>;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  isLoading: boolean;
}

interface FilterState {
  user_id: string;
  alias: string;
  blocked: string;
  region: string;
}

const CustomersView: React.FC<CustomersViewProps> = ({
  customers,
  setCustomers,
  accessToken,
  userID,
  userRole,
  isLoading,
}) => {
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<FilterState>({
    user_id: "",
    alias: "",
    blocked: "",
    region: "",
  });

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [selectedCustomer, setSelectedCustomer] = useState<Customer | null>(null);
  const [selectedCustomerId, setSelectedCustomerId] = useState<string | null>(null);
  const [customerDetailDefaultTab, setCustomerDetailDefaultTab] = useState<"overview" | "settings">("overview");
  const [lastRefreshed, setLastRefreshed] = useState(new Date().toLocaleString("en-US"));

  const handleCreateCustomer = async () => {
    if (!accessToken) return;
    try {
      const { allEndUsersCall } = await import("@/components/networking");
      const listData = await allEndUsersCall(accessToken);
      if (listData) {
        setCustomers(Array.isArray(listData) ? listData : []);
      }
      setShowCreateModal(false);
    } catch (error) {
      console.error("Error refreshing customers:", error);
    }
  };

  const handleEditCustomer = (customer: Customer) => {
    setSelectedCustomerId(customer.user_id);
    setSelectedCustomer(customer);
    setCustomerDetailDefaultTab("settings");
  };

  const handleViewInfo = (customer: Customer) => {
    setSelectedCustomerId(customer.user_id);
    setSelectedCustomer(customer);
    setCustomerDetailDefaultTab("overview");
  };

  const handleCloseCustomerInfo = () => {
    setSelectedCustomerId(null);
    setSelectedCustomer(null);
  };

  const handleUpdateCustomer = (updated: Customer) => {
    setCustomers((prev) =>
      prev.map((c) => (c.user_id === updated.user_id ? updated : c))
    );
    setSelectedCustomer(updated);
  };

  const handleDeleteClick = (customer: Customer) => {
    setSelectedCustomer(customer);
    setShowDeleteModal(true);
  };

  const handleConfirmDelete = async () => {
    if (!selectedCustomer || !accessToken) return;

    try {
      await customerDeleteCall(accessToken, selectedCustomer.user_id);
      setCustomers(
        customers.filter((c) => c.user_id !== selectedCustomer.user_id)
      );
      setShowDeleteModal(false);
      setSelectedCustomer(null);
    } catch (error) {
      console.error("Error deleting customer:", error);
    }
  };

  const handleRefresh = () => {
    setLastRefreshed(new Date().toLocaleString("en-US"));
    // Trigger a re-fetch if needed
  };

  const handleFilterChange = (key: keyof FilterState, value: string) => {
    setFilters({ ...filters, [key]: value });
  };

  const handleFilterReset = () => {
    setFilters({
      user_id: "",
      alias: "",
      blocked: "",
      region: "",
    });
  };

  const filteredCustomers = customers.filter((c) => {
    const matchesUserId = !filters.user_id || c.user_id.toLowerCase().includes(filters.user_id.toLowerCase());
    const matchesAlias = !filters.alias || (c.alias && c.alias.toLowerCase().includes(filters.alias.toLowerCase()));
    const matchesBlocked =
      !filters.blocked ||
      (filters.blocked === "active" && !c.blocked) ||
      (filters.blocked === "blocked" && c.blocked);
    const matchesRegion = !filters.region || c.allowed_model_region === filters.region;

    return matchesUserId && matchesAlias && matchesBlocked && matchesRegion;
  });

  const initialCustomerForDetail =
    selectedCustomerId && selectedCustomer?.user_id === selectedCustomerId
      ? selectedCustomer
      : customers.find((c) => c.user_id === selectedCustomerId) ?? null;

  if (selectedCustomerId) {
    return (
      <div className="w-full max-w-full px-4 py-4 md:px-6">
        <CustomerInfo
          customerId={selectedCustomerId}
          initialCustomer={initialCustomerForDetail}
          onClose={handleCloseCustomerInfo}
          onUpdate={handleUpdateCustomer}
          accessToken={accessToken}
          userRole={userRole}
          defaultTab={customerDetailDefaultTab}
        />
      </div>
    );
  }

  return (
    <div className="w-full max-w-full px-4 py-4 md:px-6">
      <div className="flex flex-col gap-4">
        {(userRole === "Admin" || userRole === "Org Admin") && (
          <Button className="w-fit" onClick={() => setShowCreateModal(true)}>
            + Create New Customer
          </Button>
        )}
        <Text className="text-sm text-gray-500">
          Customers are end-users of an AI application (e.g. users of your internal chat UI).
        </Text>

        <CustomersHeaderTabs lastRefreshed={lastRefreshed} onRefresh={handleRefresh} userRole={userRole}>
          <TabPanel>
            <Text className="block mb-3">
              Click on &ldquo;Customer ID&rdquo; to view customer details and manage settings.
            </Text>
            <Card className="w-full overflow-hidden flex flex-col min-h-[400px]">
              <div className="border-b px-4 sm:px-6 py-4 shrink-0">
                <CustomersFilters
                  filters={filters}
                  showFilters={showFilters}
                  onToggleFilters={setShowFilters}
                  onChange={(key, value) => handleFilterChange(key as keyof FilterState, value)}
                  onReset={handleFilterReset}
                />
              </div>
              <div className="overflow-auto flex-1 min-h-0">
                <CustomersTable
                  customers={filteredCustomers}
                  userRole={userRole}
                  onEdit={handleEditCustomer}
                  onDelete={handleDeleteClick}
                  onViewInfo={handleViewInfo}
                  isLoading={isLoading}
                />
              </div>
            </Card>
          </TabPanel>
        </CustomersHeaderTabs>

          {(userRole === "Admin" || userRole === "Org Admin") && (
            <>
              <CreateCustomerModal
                isOpen={showCreateModal}
                onClose={() => setShowCreateModal(false)}
                onCreate={handleCreateCustomer}
              />
              <DeleteCustomerModal
                isOpen={showDeleteModal}
                onClose={() => {
                  setShowDeleteModal(false);
                  setSelectedCustomer(null);
                }}
                onConfirm={handleConfirmDelete}
                customerName={selectedCustomer?.alias || ""}
                customerId={selectedCustomer?.user_id || ""}
              />
            </>
          )}
      </div>
    </div>
  );
};

export default CustomersView;
