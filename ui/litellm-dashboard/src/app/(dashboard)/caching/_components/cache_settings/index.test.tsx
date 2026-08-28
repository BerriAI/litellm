import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CacheSettings from "./index";

const { getCacheSettingsCall, testCacheConnectionCall, updateCacheSettingsCall } = vi.hoisted(() => ({
  getCacheSettingsCall: vi.fn(),
  testCacheConnectionCall: vi.fn(),
  updateCacheSettingsCall: vi.fn(),
}));

vi.mock("@/components/networking", () => ({
  getCacheSettingsCall,
  testCacheConnectionCall,
  updateCacheSettingsCall,
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

const renderSettings = () => render(<CacheSettings accessToken="sk-test" userRole="Admin" userID="u1" />);

describe("CacheSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getCacheSettingsCall.mockResolvedValue({ current_values: {} });
    updateCacheSettingsCall.mockResolvedValue({ status: "success" });
    testCacheConnectionCall.mockResolvedValue({ status: "success" });
  });

  it("should render the connection fields once current values load", async () => {
    renderSettings();
    expect(await screen.findByText("Connection Settings")).toBeInTheDocument();
  });

  describe("when the redis type is node", () => {
    it("should show the connection fields and hide cluster/sentinel/semantic fields", async () => {
      renderSettings();

      expect(await screen.findByText("Redis URL")).toBeInTheDocument();
      expect(screen.getByText("Database Index")).toBeInTheDocument();
      expect(screen.queryByText("Startup Nodes")).not.toBeInTheDocument();
      expect(screen.queryByText("Sentinel Nodes")).not.toBeInTheDocument();
      expect(screen.queryByText("Embedding Model")).not.toBeInTheDocument();
    });
  });

  describe("when the redis type is cluster", () => {
    it("should reveal the cluster startup nodes field", async () => {
      getCacheSettingsCall.mockResolvedValue({ current_values: { redis_type: "cluster" } });
      renderSettings();
      expect(await screen.findByText("Startup Nodes")).toBeInTheDocument();
    });
  });

  describe("when the redis type is sentinel", () => {
    it("should reveal the sentinel fields", async () => {
      getCacheSettingsCall.mockResolvedValue({ current_values: { redis_type: "sentinel" } });
      renderSettings();
      expect(await screen.findByText("Sentinel Nodes")).toBeInTheDocument();
      expect(screen.getByText("Service Name")).toBeInTheDocument();
    });
  });

  describe("when the redis type is semantic", () => {
    it("should reveal the semantic fields", async () => {
      getCacheSettingsCall.mockResolvedValue({ current_values: { redis_type: "semantic" } });
      renderSettings();
      expect(await screen.findByText("Similarity Threshold")).toBeInTheDocument();
      expect(screen.getByText("Embedding Model")).toBeInTheDocument();
    });
  });

  describe("when a field fails inline validation", () => {
    it("should block save and surface the validation message", async () => {
      const user = userEvent.setup();
      renderSettings();

      const port = await screen.findByLabelText("Port");
      await user.clear(port);
      fireEvent.change(port, { target: { value: "99999" } });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      expect(await screen.findByText(/Port must be an integer between 1 and 65535/i)).toBeInTheDocument();
      expect(updateCacheSettingsCall).not.toHaveBeenCalled();
    });

    it("should block save when a list field holds malformed JSON instead of silently dropping it", async () => {
      const user = userEvent.setup();
      getCacheSettingsCall.mockResolvedValue({ current_values: { redis_type: "cluster" } });
      renderSettings();

      const startupNodes = await screen.findByLabelText("Startup Nodes");
      fireEvent.change(startupNodes, { target: { value: "not json" } });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      expect(await screen.findByText(/Must be a valid JSON array/i)).toBeInTheDocument();
      expect(updateCacheSettingsCall).not.toHaveBeenCalled();
    });

    it("should block save with an error when a non-numeric value is entered into a numeric field", async () => {
      const user = userEvent.setup();
      renderSettings();

      const db = await screen.findByLabelText("Database Index");
      fireEvent.change(db, { target: { value: "redis://host:6379/1" } });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      expect(await screen.findByText(/Must be a non-negative integer/i)).toBeInTheDocument();
      expect(updateCacheSettingsCall).not.toHaveBeenCalled();
    });
  });

  describe("when saving a valid node configuration", () => {
    it("should send the backend payload shape with type redis and no UI-only fields", async () => {
      const user = userEvent.setup();
      renderSettings();

      const host = await screen.findByLabelText("Host");
      fireEvent.change(host, { target: { value: "localhost" } });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() =>
        expect(updateCacheSettingsCall).toHaveBeenCalledWith("sk-test", {
          type: "redis",
          host: "localhost",
          port: "6379",
          ssl: false,
          ssl_check_hostname: false,
        }),
      );
    });

    it("should include a numeric field like Database Index in the save payload", async () => {
      const user = userEvent.setup();
      renderSettings();

      fireEvent.change(await screen.findByLabelText("Redis URL"), { target: { value: "redis://host:6379/1" } });
      fireEvent.change(await screen.findByLabelText("Database Index"), { target: { value: "2" } });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalled());
      expect(updateCacheSettingsCall.mock.calls[0][1]).toMatchObject({ db: 2, url: "redis://host:6379/1" });
    });
  });
  describe("when a field is set in config.yaml", () => {
    const configSourcedResponse = {
      current_values: { type: "redis", host: "localhost", namespace: "yamlns", ttl: 1234 },
      config_sourced_fields: ["namespace", "ttl"],
    };

    it("should render the config-sourced fields disabled and explain where they come from", async () => {
      getCacheSettingsCall.mockResolvedValue(configSourcedResponse);
      const user = userEvent.setup();
      renderSettings();

      await user.click(await screen.findByText("Advanced Settings"));

      expect(await screen.findByLabelText("Namespace")).toBeDisabled();
      expect(screen.getByLabelText("TTL (seconds)")).toBeDisabled();
      expect(screen.getByLabelText("Host")).toBeEnabled();
      expect(
        screen.getAllByText("Set in config.yaml. Change it there, or remove it to edit this here.").length,
      ).toBeGreaterThan(0);
    });

    it("should exclude config-sourced fields from the save payload so a save cannot clear them", async () => {
      getCacheSettingsCall.mockResolvedValue(configSourcedResponse);
      const user = userEvent.setup();
      renderSettings();

      fireEvent.change(await screen.findByLabelText("Host"), { target: { value: "newhost" } });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalled());
      const payload = updateCacheSettingsCall.mock.calls[0][1];
      expect(payload).toMatchObject({ host: "newhost" });
      expect(payload).not.toHaveProperty("namespace");
      expect(payload).not.toHaveProperty("ttl");
    });

    it("should omit type from the save payload when config.yaml owns it", async () => {
      getCacheSettingsCall.mockResolvedValue({
        current_values: { type: "redis", host: "localhost", namespace: "yamlns" },
        config_sourced_fields: ["type", "namespace"],
      });
      const user = userEvent.setup();
      renderSettings();

      fireEvent.change(await screen.findByLabelText("Host"), { target: { value: "newhost" } });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalled());
      expect(updateCacheSettingsCall.mock.calls[0][1]).not.toHaveProperty("type");
    });

    it("should still let the deployment topology be switched when config.yaml only pins type redis", async () => {
      getCacheSettingsCall.mockResolvedValue({
        current_values: { type: "redis", host: "localhost" },
        config_sourced_fields: ["type"],
      });
      const user = userEvent.setup();
      renderSettings();

      await user.click(await screen.findByRole("combobox"));

      expect(await screen.findByRole("option", { name: "Semantic" })).toHaveAttribute("data-disabled");
      expect(screen.getByRole("option", { name: "Cluster" })).not.toHaveAttribute("data-disabled");

      await user.click(screen.getByRole("option", { name: "Cluster" }));
      expect(await screen.findByLabelText("Startup Nodes")).toBeInTheDocument();
    });

    it("should still send config-sourced fields when testing the connection", async () => {
      getCacheSettingsCall.mockResolvedValue(configSourcedResponse);
      const user = userEvent.setup();
      renderSettings();

      await user.click(await screen.findByRole("button", { name: /test connection/i }));

      await waitFor(() => expect(testCacheConnectionCall).toHaveBeenCalled());
      expect(testCacheConnectionCall.mock.calls[0][1]).toMatchObject({ namespace: "yamlns", ttl: 1234 });
    });
  });
});
