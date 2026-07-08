import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
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
      await user.type(port, "99999");
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      expect(await screen.findByText(/Port must be an integer between 1 and 65535/i)).toBeInTheDocument();
      expect(updateCacheSettingsCall).not.toHaveBeenCalled();
    });

    it("should block save when a list field holds malformed JSON instead of silently dropping it", async () => {
      const user = userEvent.setup();
      getCacheSettingsCall.mockResolvedValue({ current_values: { redis_type: "cluster" } });
      renderSettings();

      const startupNodes = await screen.findByLabelText("Startup Nodes");
      await user.type(startupNodes, "not json");
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      expect(await screen.findByText(/Must be a valid JSON array/i)).toBeInTheDocument();
      expect(updateCacheSettingsCall).not.toHaveBeenCalled();
    });

    it("should block save with an error when a non-numeric value is entered into a numeric field", async () => {
      const user = userEvent.setup();
      renderSettings();

      const db = await screen.findByLabelText("Database Index");
      await user.type(db, "redis://host:6379/1");
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
      await user.type(host, "localhost");
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

      await user.type(await screen.findByLabelText("Redis URL"), "redis://host:6379/1");
      await user.type(await screen.findByLabelText("Database Index"), "2");
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalled());
      expect(updateCacheSettingsCall.mock.calls[0][1]).toMatchObject({ db: 2, url: "redis://host:6379/1" });
    });
  });
});
