"use client";

import { Contact, Plus } from "lucide-react";
import { parseAsBoolean, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback, useState } from "react";

import {
  useCustomer,
  useDeleteCustomers,
  useSetCustomerBlockedState,
  useSetCustomersBlockedState,
} from "@/app/(dashboard)/hooks/customers/useCustomerMutations";
import { useCustomers, type EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { Alert, AlertTitle } from "@/components/shared/Alert";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderControls, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";
import { all_admin_roles, isProxyAdminRole } from "@/utils/roles";

import { CreateCustomerModal } from "./CreateCustomerModal";
import CustomerInfoView from "./CustomerInfoView";
import CustomersTable from "./CustomersTable";
import { hasSharedCustomerBudget } from "./customerPayload";

const CUSTOMER_VIEW_PARSERS = {
  customer: parseAsString,
  edit: parseAsBoolean.withDefault(false),
  customer_tab: parseAsStringLiteral(["overview", "settings"]).withDefault("overview"),
};

function CustomersHeader({ canModify, onCreate }: { canModify: boolean; onCreate: () => void }) {
  return (
    <PageHeader>
      <PageHeaderTitle>
        <Contact />
        Customers
      </PageHeaderTitle>
      <PageHeaderDescription>Manage end-user spend, budgets, rate limits, model and MCP access</PageHeaderDescription>
      {canModify && (
        <PageHeaderControls>
          <Button onClick={onCreate}>
            <Plus />
            Create Customer
          </Button>
        </PageHeaderControls>
      )}
    </PageHeader>
  );
}

export default function CustomersPanel() {
  const { accessToken, userRole, isViewOnly, premiumUser } = useAuthorized();
  const canModify = isProxyAdminRole(userRole ?? "") && !isViewOnly;
  const [{ customer: customerId, edit, customer_tab: tab }, setView] = useQueryStates(CUSTOMER_VIEW_PARSERS, {
    history: "push",
  });
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<EndUser | null>(null);
  const customers = useCustomers();
  const customer = useCustomer(customerId);
  const deleteCustomers = useDeleteCustomers();
  const setBlockedState = useSetCustomerBlockedState();
  const setBulkBlockedState = useSetCustomersBlockedState();
  const isBlocking = setBlockedState.isPending || setBulkBlockedState.isPending;

  const openCustomer = useCallback(
    (id: string) => void setView({ customer: id, edit: null, customer_tab: null }),
    [setView],
  );
  const editCustomer = useCallback(
    (id: string) => void setView({ customer: id, edit: true, customer_tab: "settings" }),
    [setView],
  );
  const closeCustomer = useCallback(() => void setView({ customer: null, edit: null, customer_tab: null }), [setView]);
  const cancelEdit = useCallback(() => void setView({ edit: null, customer_tab: "settings" }), [setView]);
  const handleSetBlocked = useCallback(
    async (target: EndUser) => {
      const canSetBlocked = canModify && !isBlocking && premiumUser;
      if (!canSetBlocked) return;
      try {
        await setBlockedState.mutateAsync({ userId: target.user_id, blocked: !target.blocked });
        toast.success(target.blocked ? "Customer unblocked" : "Customer blocked");
      } catch (error) {
        toast.fromError(error);
      }
    },
    [canModify, premiumUser, isBlocking, setBlockedState],
  );
  const handleDelete = async () => {
    if (!deleteTarget || !canModify) return;
    try {
      await deleteCustomers.mutateAsync([deleteTarget.user_id]);
      if (customerId === deleteTarget.user_id) closeCustomer();
      setDeleteTarget(null);
      toast.success("Customer deleted");
    } catch (error) {
      toast.fromError(error);
    }
  };

  const handleBulkSetBlocked = async (userIds: string[], blocked: boolean): Promise<boolean> => {
    const canSetBlocked = canModify && !isBlocking && premiumUser;
    if (!userIds.length || !canSetBlocked) return false;
    try {
      await setBulkBlockedState.mutateAsync({ userIds, blocked });
      toast.success(
        `${userIds.length} customer${userIds.length === 1 ? "" : "s"} ${blocked ? "blocked" : "unblocked"}`,
      );
      return true;
    } catch (error) {
      toast.fromError(error);
      return false;
    }
  };

  if (!all_admin_roles.includes(userRole ?? "")) {
    return (
      <Page>
        <Alert>
          <AlertTitle>You need an admin role to view customers</AlertTitle>
        </Alert>
      </Page>
    );
  }

  return (
    <Page className={customerId ? "h-full overflow-y-auto" : "h-full overflow-hidden"}>
      {customerId ? (
        <CustomerDetail
          customer={customer.data}
          isLoading={customer.isLoading}
          isError={customer.isError}
          accessToken={accessToken}
          canModify={canModify}
          premiumUser={premiumUser}
          isEdit={edit}
          hasSharedBudget={
            customer.data && customers.data && !customers.isError
              ? hasSharedCustomerBudget(customer.data, customers.data)
              : undefined
          }
          tab={tab}
          onTabChange={(value) => void setView({ customer_tab: value, edit: null })}
          isBlocking={isBlocking}
          onBack={closeCustomer}
          onEdit={() => editCustomer(customerId)}
          onCancelEdit={cancelEdit}
          onDelete={setDeleteTarget}
          onSetBlocked={handleSetBlocked}
        />
      ) : (
        <>
          <CustomersHeader canModify={canModify} onCreate={() => setIsCreateOpen(true)} />
          <CustomersTable
            isBlocking={isBlocking}
            onBulkSetBlocked={handleBulkSetBlocked}
            data={customers.data ?? []}
            isLoading={customers.isLoading}
            isError={customers.isError}
            onRefresh={() => void customers.refetch()}
            isRefreshing={customers.isFetching}
            canModify={canModify}
            premiumUser={premiumUser}
            onView={openCustomer}
            onEdit={editCustomer}
            onDelete={setDeleteTarget}
            onSetBlocked={handleSetBlocked}
          />
        </>
      )}
      {canModify && (
        <>
          <CreateCustomerModal open={isCreateOpen} onOpenChange={setIsCreateOpen} onCreated={openCustomer} />
          <DeleteResourceModal
            isOpen={deleteTarget !== null}
            title="Delete Customer?"
            message="A later request naming the same end user recreates the customer with no restrictions"
            resourceInformationTitle="Customer Information"
            resourceInformation={[{ label: "Customer ID", value: deleteTarget?.user_id, code: true }]}
            requiredConfirmation={deleteTarget?.user_id}
            onCancel={() => setDeleteTarget(null)}
            onOk={handleDelete}
            confirmLoading={deleteCustomers.isPending}
          />
        </>
      )}
    </Page>
  );
}

type CustomerDetailProps = Omit<Parameters<typeof CustomerInfoView>[0], "customer"> & {
  customer: EndUser | undefined;
  isLoading: boolean;
  isError: boolean;
};

function CustomerDetail({ customer, isLoading, isError, onBack, ...props }: CustomerDetailProps) {
  if (isLoading) return <p role="status">Loading customer</p>;
  if (isError || !customer) {
    return (
      <>
        <Button variant="outline" onClick={onBack}>
          Back to Customers
        </Button>
        <Alert variant="error">
          <AlertTitle>Unable to load customer</AlertTitle>
        </Alert>
      </>
    );
  }
  return <CustomerInfoView key={customer.user_id} {...props} customer={customer} onBack={onBack} />;
}
