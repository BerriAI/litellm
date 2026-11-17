import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import CloudZeroExportModal from "./cloudzero_export_modal";

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), fromError: vi.fn(), info: vi.fn() },
}));

vi.mock("@/components/networking", () => ({
  getGlobalLitellmHeaderName: () => "Authorization",
}));

const jsonResponse = (status: number, body: unknown) =>
  ({ ok: status >= 200 && status < 300, status, json: async () => body }) as Response;

const bodyOf = (call: unknown[]) => JSON.parse((call[1] as RequestInit).body as string);

const callsTo = (fetchMock: ReturnType<typeof vi.fn>, path: string) =>
  fetchMock.mock.calls.filter((call) => call[0] === path);

describe("CloudZeroExportModal", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn(async (url: string) => {
      if (url === "/cloudzero/settings") return jsonResponse(404, { error: "not configured" });
      if (url === "/cloudzero/init") return jsonResponse(200, { message: "saved" });
      if (url === "/cloudzero/export") return jsonResponse(200, { message: "exported" });
      return jsonResponse(500, { error: "unexpected" });
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  const open = () => render(<CloudZeroExportModal isOpen onClose={vi.fn()} accessToken="sk-test" />);

  it("saves the settings with a hardcoded UTC timezone before exporting", async () => {
    const user = userEvent.setup();
    open();

    fireEvent.change(await screen.findByLabelText("CloudZero API Key"), { target: { value: "cz-key-123" } });
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "conn-abc" } });
    await user.click(screen.getByRole("button", { name: "Export to CloudZero" }));

    await waitFor(() => expect(callsTo(fetchMock, "/cloudzero/init")).toHaveLength(1));

    const [initCall] = callsTo(fetchMock, "/cloudzero/init");
    expect((initCall[1] as RequestInit).method).toBe("POST");
    expect(bodyOf(initCall)).toEqual({
      api_key: "cz-key-123",
      connection_id: "conn-abc",
      timezone: "UTC",
    });
  });

  it("exports with a fixed limit and operation once the settings are saved", async () => {
    const user = userEvent.setup();
    open();

    fireEvent.change(await screen.findByLabelText("CloudZero API Key"), { target: { value: "cz-key-123" } });
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "conn-abc" } });
    await user.click(screen.getByRole("button", { name: "Export to CloudZero" }));

    await waitFor(() => expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(1));

    const [exportCall] = callsTo(fetchMock, "/cloudzero/export");
    expect((exportCall[1] as RequestInit).method).toBe("POST");
    expect(bodyOf(exportCall)).toEqual({ limit: 100000, operation: "replace_hourly" });
  });

  it("sends nothing while a required field is still empty", async () => {
    const user = userEvent.setup();
    open();

    fireEvent.change(await screen.findByLabelText("CloudZero API Key"), { target: { value: "cz-key-123" } });
    await user.click(screen.getByRole("button", { name: "Export to CloudZero" }));

    await screen.findByText("Please enter the CloudZero connection ID");
    expect(callsTo(fetchMock, "/cloudzero/init")).toHaveLength(0);
    expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(0);
  });

  it("reports both required messages when the form is untouched", async () => {
    const user = userEvent.setup();
    open();

    await screen.findByLabelText("CloudZero API Key");
    await user.click(screen.getByRole("button", { name: "Export to CloudZero" }));

    expect(await screen.findByText("Please enter your CloudZero API key")).toBeInTheDocument();
    expect(screen.getByText("Please enter the CloudZero connection ID")).toBeInTheDocument();
    expect(callsTo(fetchMock, "/cloudzero/init")).toHaveLength(0);
  });

  it("skips the save entirely when CloudZero is already configured", async () => {
    fetchMock.mockImplementation(async (url: string) => {
      if (url === "/cloudzero/settings")
        return jsonResponse(200, {
          api_key_masked: "cz-1****4567",
          connection_id: "conn-existing",
          status: "configured",
        });
      if (url === "/cloudzero/export") return jsonResponse(200, { message: "exported" });
      return jsonResponse(500, { error: "unexpected" });
    });
    const user = userEvent.setup();
    open();

    await screen.findByText(/conn-existing/);
    expect(screen.queryByLabelText("CloudZero API Key")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Export to CloudZero" }));

    await waitFor(() => expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(1));
    expect(callsTo(fetchMock, "/cloudzero/init")).toHaveLength(0);
  });

  it("masks the API key it echoes back after a successful save", async () => {
    const user = userEvent.setup();
    open();

    fireEvent.change(await screen.findByLabelText("CloudZero API Key"), { target: { value: "cz-key-123456789" } });
    fireEvent.change(screen.getByLabelText("Connection ID"), { target: { value: "conn-abc" } });
    await user.click(screen.getByRole("button", { name: "Export to CloudZero" }));

    await waitFor(() => expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(1));
    expect(bodyOf(callsTo(fetchMock, "/cloudzero/init")[0]).api_key).toBe("cz-key-123456789");
  });

  it("switches to the CSV destination without touching the CloudZero endpoints", async () => {
    const user = userEvent.setup();
    open();

    await screen.findByLabelText("CloudZero API Key");
    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: "Export to CSV" }));

    expect(screen.queryByLabelText("CloudZero API Key")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Export CSV" }));

    expect(callsTo(fetchMock, "/cloudzero/init")).toHaveLength(0);
    expect(callsTo(fetchMock, "/cloudzero/export")).toHaveLength(0);
  });

  it("keeps the API key entry obscured", async () => {
    open();
    expect(await screen.findByLabelText("CloudZero API Key")).toHaveAttribute("type", "password");
  });
});
