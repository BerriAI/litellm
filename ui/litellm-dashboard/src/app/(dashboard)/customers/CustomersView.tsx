import React, { useState } from "react";
import { Button, Card, Col, Grid, TabPanel, Text } from "@tremor/react";
import { RefreshCw } from "lucide-react";
import CustomersTable from "@/app/(dashboard)/customers/components/CustomersTable";
import CreateCustomerModal from "@/app/(dashboard)/customers/components/modals/CreateCustomerModal";
import CustomerInfoModal from "@/app/(dashboard)/customers/components/modals/CustomerInfoModal";
import DeleteCustomerModal from "@/app/(dashboard)/customers/components/modals/DeleteCustomerModal";
import CustomersHeaderTabs from "@/app/(dashboard)/customers/components/CustomersHeaderTabs";
import CustomersFilters from "@/app/(dashboard)/customers/components/CustomersFilters";
import type { Customer, NewCustomerData } from "@/app/(dashboard)/customers/types";
import { customerCreateCall, customerDeleteCall, customerUpdateCall } from "@/components/networking";

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
  const [showInfoModal, setShowInfoModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [selectedCustomer, setSelectedCustomer] = useState<Customer | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState(new Date().toLocaleString("en-US"));

  const handleCreateCustomer = async (data: NewCustomerData) => {
    if (!accessToken) return;

    try {
      const response = await customerCreateCall(accessToken, data);
      if (response) {
        // Refresh the customer list
        const { allEndUsersCall } = await import("@/components/networking");
        const listData = await allEndUsersCall(accessToken);
        if (listData) {
          setCustomers(Array.isArray(listData) ? listData : []);
        }
      }
      setShowCreateModal(false);
    } catch (error) {
      console.error("Error creating customer:", error);
    }
  };

  const handleEditCustomer = (customer: Customer) => {
    setSelectedCustomer(customer);
    setShowInfoModal(true);
  };

  const handleViewInfo = (customer: Customer) => {
    setSelectedCustomer(customer);
    setShowInfoModal(true);
  };

  const handleSaveCustomer = async (updated: Customer) => {
    if (!accessToken) return;

    try {
      await customerUpdateCall(accessToken, updated);
      setCustomers(
        customers.map((c) => (c.user_id === updated.user_id ? updated : c))
      );
      setShowInfoModal(false);
      setSelectedCustomer(null);
    } catch (error) {
      console.error("Error updating customer:", error);
    }
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

  return (
    <div className="w-full mx-4 h-[75vh]">
      <Grid numItems={1} className="gap-2 p-8 w-full mt-2">
        <Col numColSpan={1} className="flex flex-col gap-2">
          {(userRole === "Admin" || userRole === "Org Admin") && (
            <Button className="w-fit" onClick={() => setShowCreateModal(true)}>
              + Create New Customer
            </Button>
          )}

          <CustomersHeaderTabs lastRefreshed={lastRefreshed} onRefresh={handleRefresh} userRole={userRole}>
            <TabPanel>
              <Text>
                Click on &ldquo;Customer ID&rdquo; to view customer details and manage settings.
              </Text>
              <Grid numItems={1} className="gap-2 pt-2 pb-2 h-[75vh] w-full mt-2">
                <Col numColSpan={1}>
                  <Card className="w-full mx-auto flex-auto overflow-hidden overflow-y-auto max-h-[50vh]">
                    <div className="border-b px-6 py-4">
                      <CustomersFilters
                        filters={filters}
                        showFilters={showFilters}
                        onToggleFilters={setShowFilters}
                        onChange={handleFilterChange}
                        onReset={handleFilterReset}
                      />
                    </div>
                    <CustomersTable
                      customers={filteredCustomers}
                      userRole={userRole}
                      onEdit={handleEditCustomer}
                      onDelete={handleDeleteClick}
                      onViewInfo={handleViewInfo}
                      isLoading={isLoading}
                    />
                  </Card>
                </Col>
              </Grid>
            </TabPanel>
          </CustomersHeaderTabs>

          {(userRole === "Admin" || userRole === "Org Admin") && (
            <>
              <CreateCustomerModal
                isOpen={showCreateModal}
                onClose={() => setShowCreateModal(false)}
                onCreate={handleCreateCustomer}
              />
              <CustomerInfoModal
                isOpen={showInfoModal}
                onClose={() => {
                  setShowInfoModal(false);
                  setSelectedCustomer(null);
                }}
                customer={selectedCustomer}
                onSave={handleSaveCustomer}
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
        </Col>
      </Grid>
    </div>
  );
};

export default CustomersView;
