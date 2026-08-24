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

const LOADED_WITH_ADVANCED = {
  current_values: {
    host: "redis.internal",
    namespace: "prod-ns",
    ttl: "300",
    max_connections: "50",
    ssl: true,
  },
};

const renderSettings = () => render(<CacheSettings accessToken="sk-test" userRole="Admin" userID="u1" />);

const save = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: /save changes/i }));

describe("CacheSettings advanced settings round-trip", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getCacheSettingsCall.mockResolvedValue(LOADED_WITH_ADVANCED);
    updateCacheSettingsCall.mockResolvedValue({ status: "success" });
    testCacheConnectionCall.mockResolvedValue({ status: "success" });
  });

  it("keeps loaded advanced values in the payload when the section is never opened", async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByText("Connection Settings");

    expect(screen.queryByLabelText("Namespace")).not.toBeInTheDocument();
    await save(user);

    await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalledTimes(1));
    expect(updateCacheSettingsCall.mock.calls[0][1]).toEqual({
      type: "redis",
      host: "redis.internal",
      port: "6379",
      ssl: true,
      ssl_check_hostname: false,
      namespace: "prod-ns",
      ttl: 300,
      max_connections: 50,
    });
  });

  it("reveals the advanced field sections only after the user expands them", async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByText("Connection Settings");
    expect(screen.queryByText("SSL Settings")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Advanced Settings" }));

    expect(await screen.findByText("SSL Settings")).toBeInTheDocument();
    expect(screen.getByText("Cache Management")).toBeInTheDocument();
    expect(screen.getByText("GCP Authentication")).toBeInTheDocument();
  });

  it("sends the same payload whether or not the advanced section was expanded", async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByText("Connection Settings");
    await save(user);
    await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalledTimes(1));
    const whileCollapsed = updateCacheSettingsCall.mock.calls[0][1];

    await user.click(screen.getByText("Advanced Settings"));
    await screen.findByLabelText("Namespace");
    await save(user);

    await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalledTimes(2));
    expect(updateCacheSettingsCall.mock.calls[1][1]).toEqual(whileCollapsed);
  });

  it("preserves a value typed into the advanced section after it is collapsed again", async () => {
    getCacheSettingsCall.mockResolvedValue({ current_values: { host: "redis.internal" } });
    const user = userEvent.setup();
    renderSettings();
    await screen.findByText("Connection Settings");

    await user.click(screen.getByText("Advanced Settings"));
    fireEvent.change(await screen.findByLabelText("Namespace"), { target: { value: "typed-ns" } });
    await user.click(screen.getByText("Advanced Settings"));
    await waitFor(() => expect(screen.queryByLabelText("Namespace")).not.toBeInTheDocument());

    await save(user);

    await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalledTimes(1));
    expect(updateCacheSettingsCall.mock.calls[0][1]).toMatchObject({ namespace: "typed-ns" });
  });

  it("restores a value typed into the advanced section when it is expanded again", async () => {
    getCacheSettingsCall.mockResolvedValue({ current_values: { host: "redis.internal" } });
    const user = userEvent.setup();
    renderSettings();
    await screen.findByText("Connection Settings");

    await user.click(screen.getByText("Advanced Settings"));
    fireEvent.change(await screen.findByLabelText("Namespace"), { target: { value: "typed-ns" } });
    await user.click(screen.getByText("Advanced Settings"));
    await waitFor(() => expect(screen.queryByLabelText("Namespace")).not.toBeInTheDocument());
    await user.click(screen.getByText("Advanced Settings"));

    expect(await screen.findByLabelText("Namespace")).toHaveValue("typed-ns");
  });

  it("keeps a hidden section's fields out of the payload when the redis type does not use them", async () => {
    getCacheSettingsCall.mockResolvedValue({
      current_values: { redis_type: "sentinel", service_name: "mymaster", host: "redis.internal" },
    });
    const user = userEvent.setup();
    renderSettings();
    await screen.findByLabelText("Service Name");
    await save(user);

    await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalledTimes(1));
    expect(updateCacheSettingsCall.mock.calls[0][1]).toMatchObject({ service_name: "mymaster" });
    expect(updateCacheSettingsCall.mock.calls[0][1]).not.toHaveProperty("redis_startup_nodes");
  });

  it("does not block the save on a malformed value inside a collapsed advanced section", async () => {
    getCacheSettingsCall.mockResolvedValue({ current_values: { host: "redis.internal" } });
    const user = userEvent.setup();
    renderSettings();
    await screen.findByText("Connection Settings");

    await user.click(screen.getByText("Advanced Settings"));
    fireEvent.change(await screen.findByLabelText("TTL (seconds)"), { target: { value: "not-a-number" } });
    await user.click(screen.getByText("Advanced Settings"));
    await waitFor(() => expect(screen.queryByLabelText("TTL (seconds)")).not.toBeInTheDocument());

    await save(user);

    await waitFor(() => expect(updateCacheSettingsCall).toHaveBeenCalledTimes(1));
    expect(updateCacheSettingsCall.mock.calls[0][1]).not.toHaveProperty("ttl");
  });
});
