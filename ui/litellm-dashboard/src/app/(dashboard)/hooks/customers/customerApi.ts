import { apiClient } from "@/components/networking";
import type { components } from "@/lib/http/schema";
import type { EndUser } from "./useCustomers";

type Schemas = components["schemas"];

export type CreateCustomerRequest = Partial<Schemas["NewCustomerRequest"]> & {
  user_id: string;
};
export type UpdateCustomerRequest = Partial<Schemas["UpdateCustomerRequest"]> & { user_id: string };
export type BulkUpdateCustomersRequest = Omit<Schemas["BulkUpdateCustomerRequest"], "customers"> & {
  customers: UpdateCustomerRequest[];
};
export type BulkUpdateCustomersResponse = Schemas["BulkUpdateCustomerResponse"];
export type CustomerBlockedStateInput = { userId: string; blocked: boolean };
export type CustomersBlockedStateInput = {
  userIds: string[];
  blocked: boolean;
};

export const createCustomer = (accessToken: string, body: CreateCustomerRequest): Promise<EndUser> =>
  apiClient.post<EndUser>("/customer/new", { accessToken, body });

export const updateCustomer = (accessToken: string, body: UpdateCustomerRequest): Promise<EndUser> =>
  apiClient.post<EndUser>("/customer/update", { accessToken, body });

export const bulkUpdateCustomers = (
  accessToken: string,
  body: BulkUpdateCustomersRequest,
): Promise<BulkUpdateCustomersResponse> =>
  apiClient.post<BulkUpdateCustomersResponse>("/customer/bulk_update", { accessToken, body });

export const deleteCustomers = (accessToken: string, userIds: string[]): Promise<Schemas["DeleteCustomersResponse"]> =>
  apiClient.post<Schemas["DeleteCustomersResponse"]>("/customer/delete", {
    accessToken,
    body: { user_ids: userIds },
  });

export const setCustomersBlockedState = (
  accessToken: string,
  { userIds, blocked }: CustomersBlockedStateInput,
): Promise<Schemas["BlockUsersResponse"] | Schemas["UnblockUsersResponse"]> =>
  apiClient.post<Schemas["BlockUsersResponse"] | Schemas["UnblockUsersResponse"]>(
    blocked ? "/customer/block" : "/customer/unblock",
    {
      accessToken,
      body: { user_ids: userIds },
    },
  );

export const setCustomerBlockedState = (
  accessToken: string,
  { userId, blocked }: CustomerBlockedStateInput,
): Promise<Schemas["BlockUsersResponse"] | Schemas["UnblockUsersResponse"]> =>
  setCustomersBlockedState(accessToken, { userIds: [userId], blocked });
