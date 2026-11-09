import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import CloudZeroCreateModal from "./CloudZeroCreateModal";

const mutate = vi.fn();

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: () => ({ accessToken: "test-token" }),
}));

vi.mock("@/app/(dashboard)/hooks/cloudzero/useCloudZeroCreate", () => ({
  useCloudZeroCreate: () => ({ mutate, isPending: false }),
}));

const renderModal = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <CloudZeroCreateModal open={true} onOk={vi.fn()} onCancel={vi.fn()} />
    </QueryClientProvider>,
  );
};

const submittedPayload = (): Record<string, unknown> => {
  expect(mutate).toHaveBeenCalledTimes(1);
  return mutate.mock.calls[0][0] as Record<string, unknown>;
};

describe("CloudZeroCreateModal submit payload", () => {
  beforeEach(() => {
    mutate.mockClear();
  });

  it("sends every filled field verbatim", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "cz-secret-key" } });
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "conn-42" } });
    fireEvent.change(screen.getByLabelText("Timezone"), { target: { value: "America/New_York" } });
    await user.click(screen.getByRole("button", { name: "Create" }));

    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({
        connection_id: "conn-42",
        timezone: "America/New_York",
        api_key: "cz-secret-key",
      }),
    );
  });

  it("defaults an untouched timezone to UTC", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "cz-secret-key" } });
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "conn-42" } });
    await user.click(screen.getByRole("button", { name: "Create" }));

    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({
        connection_id: "conn-42",
        timezone: "UTC",
        api_key: "cz-secret-key",
      }),
    );
  });

  it("blocks submission and shows both required messages when the form is empty", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getByRole("button", { name: "Create" }));

    expect(await screen.findByText("Please enter your CloudZero API key")).toBeInTheDocument();
    expect(screen.getByText("Please enter your CloudZero connection ID")).toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("does not submit when Enter is pressed inside a text field", async () => {
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("CloudZero API Key"), { target: { value: "cz-secret-key" } });
    await user.type(screen.getByLabelText("Connection ID"), "conn-42{Enter}");

    expect(mutate).not.toHaveBeenCalled();
  });
});
