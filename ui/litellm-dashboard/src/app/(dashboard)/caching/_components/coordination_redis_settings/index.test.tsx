import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import CoordinationRedisSettings from "./index";
import { REDACTED_VALUE } from "./types";
import * as networking from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";

vi.mock("@/components/networking", () => ({
  getCoordinationRedisSettingsCall: vi.fn(),
  testCoordinationRedisConnectionCall: vi.fn(),
  updateCoordinationRedisSettingsCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test" }),
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: { success: vi.fn(), fromBackend: vi.fn() },
}));

const getSettings = vi.mocked(networking.getCoordinationRedisSettingsCall);
const updateSettings = vi.mocked(networking.updateCoordinationRedisSettingsCall);
const testConnection = vi.mocked(networking.testCoordinationRedisConnectionCall);
const notifications = vi.mocked(NotificationsManager);

const settingsResponse = (
  values: Record<string, unknown>,
  source: "coordination_redis" | "cache_backend" | "environment" | null = null,
) => ({ values, fields: [], source });

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
};

const renderSettings = () => render(<CoordinationRedisSettings />, { wrapper });

const clickSave = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: /save changes/i }));

describe("CoordinationRedisSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getSettings.mockResolvedValue(settingsResponse({}));
    updateSettings.mockResolvedValue(undefined);
    testConnection.mockResolvedValue({ status: "healthy" });
  });

  describe("when the redis type is node", () => {
    it("should show the connection fields and hide cluster/sentinel fields", async () => {
      renderSettings();

      expect(await screen.findByText("Connection Settings")).toBeInTheDocument();
      expect(screen.getByText("Redis URL")).toBeInTheDocument();
      expect(screen.getByText("SSL")).toBeInTheDocument();
      expect(screen.queryByText("Startup Nodes")).not.toBeInTheDocument();
      expect(screen.queryByText("Sentinel Nodes")).not.toBeInTheDocument();
    });

    it("should not offer semantic caching, which is a response-cache-only concern", async () => {
      renderSettings();
      await screen.findByText("Connection Settings");
      expect(screen.queryByText(/semantic/i)).not.toBeInTheDocument();
    });
  });

  describe("when the saved settings describe a cluster", () => {
    it("should reveal the cluster startup nodes field", async () => {
      getSettings.mockResolvedValue(settingsResponse({ startup_nodes: [{ host: "127.0.0.1", port: 7001 }] }));
      renderSettings();
      expect(await screen.findByText("Startup Nodes")).toBeInTheDocument();
      expect(screen.queryByText("Sentinel Nodes")).not.toBeInTheDocument();
    });
  });

  describe("when the saved settings describe a sentinel", () => {
    it("should reveal the sentinel fields", async () => {
      getSettings.mockResolvedValue(settingsResponse({ sentinel_nodes: [["localhost", 26379]] }));
      renderSettings();
      expect(await screen.findByText("Sentinel Nodes")).toBeInTheDocument();
      expect(screen.getByText("Service Name")).toBeInTheDocument();
      expect(screen.getByText("Sentinel Password")).toBeInTheDocument();
    });
  });

  describe("the source badge", () => {
    it.each([
      ["coordination_redis", "Configured here"],
      ["cache_backend", "Borrowed from response cache"],
      ["environment", "From REDIS_* environment"],
    ] as const)("should render %s as %s", async (source, label) => {
      getSettings.mockResolvedValue(settingsResponse({}, source));
      renderSettings();
      expect(await screen.findByTestId("coordination-redis-source")).toHaveTextContent(label);
    });

    it("should render a null source as not configured", async () => {
      getSettings.mockResolvedValue(settingsResponse({}, null));
      renderSettings();
      expect(await screen.findByTestId("coordination-redis-source")).toHaveTextContent("Not configured");
    });

    it("should tell the admin that saved changes need a proxy restart", async () => {
      renderSettings();
      expect(await screen.findByText(/take effect on proxy restart/i)).toBeInTheDocument();
    });
  });

  describe("when a field fails inline validation", () => {
    it("should block save and surface the port validation message", async () => {
      const user = userEvent.setup();
      renderSettings();

      const port = await screen.findByLabelText("Port");
      await user.clear(port);
      await user.type(port, "99999");
      await clickSave(user);

      expect(await screen.findByText(/Port must be an integer between 1 and 65535/i)).toBeInTheDocument();
      expect(updateSettings).not.toHaveBeenCalled();
    });

    it("should block save when a list field holds malformed JSON instead of silently dropping it", async () => {
      const user = userEvent.setup();
      getSettings.mockResolvedValue(settingsResponse({ startup_nodes: [], sentinel_nodes: [["localhost", 26379]] }));
      renderSettings();

      const sentinelNodes = await screen.findByLabelText("Sentinel Nodes");
      await user.clear(sentinelNodes);
      await user.type(sentinelNodes, "not json");
      await clickSave(user);

      expect(await screen.findByText(/Must be a valid JSON array/i)).toBeInTheDocument();
      expect(updateSettings).not.toHaveBeenCalled();
    });
  });

  describe("when saving", () => {
    it("should send a node payload with a numeric port and no empty fields", async () => {
      const user = userEvent.setup();
      renderSettings();

      await user.type(await screen.findByLabelText("Host"), "coord-redis");
      await clickSave(user);

      await waitFor(() =>
        expect(updateSettings).toHaveBeenCalledWith("sk-test", { host: "coord-redis", port: 6379, ssl: false }),
      );
    });

    it("should parse the cluster startup nodes textarea into a JSON array", async () => {
      const user = userEvent.setup();
      getSettings.mockResolvedValue(settingsResponse({ startup_nodes: [{ host: "127.0.0.1", port: 7001 }] }));
      renderSettings();

      await screen.findByLabelText("Startup Nodes");
      await clickSave(user);

      await waitFor(() => expect(updateSettings).toHaveBeenCalled());
      expect(updateSettings.mock.calls[0][1]).toMatchObject({
        startup_nodes: [{ host: "127.0.0.1", port: 7001 }],
      });
    });

    it("should not resubmit a redacted secret the admin never touched", async () => {
      const user = userEvent.setup();
      getSettings.mockResolvedValue(
        settingsResponse({ host: "coord-redis", password: REDACTED_VALUE, url: REDACTED_VALUE }),
      );
      renderSettings();

      await waitFor(() => expect(screen.getByLabelText("Host")).toHaveValue("coord-redis"));
      await clickSave(user);

      await waitFor(() => expect(updateSettings).toHaveBeenCalled());
      const payload = updateSettings.mock.calls[0][1];
      expect(payload).not.toHaveProperty("password");
      expect(payload).not.toHaveProperty("url");
      expect(payload).toMatchObject({ host: "coord-redis" });
    });

    it("should leave an already-set secret blank and say so, rather than prefilling the redacted marker", async () => {
      getSettings.mockResolvedValue(settingsResponse({ password: REDACTED_VALUE }));
      renderSettings();

      const password = await screen.findByLabelText("Password");
      await waitFor(() => expect(password).toHaveValue(""));
      expect(password).toHaveAttribute("placeholder", expect.stringMatching(/already set/i));
      expect(screen.queryByDisplayValue(REDACTED_VALUE)).not.toBeInTheDocument();
    });

    it("should submit a secret the admin typed into the blank field", async () => {
      const user = userEvent.setup();
      getSettings.mockResolvedValue(settingsResponse({ host: "coord-redis", password: REDACTED_VALUE }));
      renderSettings();

      const password = await screen.findByLabelText("Password");
      await waitFor(() => expect(password).toHaveValue(""));
      await user.type(password, "new-secret");
      await clickSave(user);

      await waitFor(() => expect(updateSettings).toHaveBeenCalled());
      expect(updateSettings.mock.calls[0][1]).toMatchObject({ password: "new-secret" });
    });

    it("should tell the admin a restart is needed once the save succeeds", async () => {
      const user = userEvent.setup();
      renderSettings();

      await screen.findByLabelText("Host");
      await clickSave(user);

      await waitFor(() => expect(notifications.success).toHaveBeenCalledWith(expect.stringMatching(/restart/i)));
    });
  });

  describe("when testing the connection", () => {
    it("should report a healthy backend response as a success", async () => {
      const user = userEvent.setup();
      renderSettings();

      await screen.findByLabelText("Host");
      await user.click(screen.getByRole("button", { name: /test connection/i }));

      await waitFor(() => expect(notifications.success).toHaveBeenCalledWith(expect.stringMatching(/successful/i)));
      expect(testConnection).toHaveBeenCalledWith("sk-test", { port: 6379, ssl: false });
    });

    it("should surface the backend error when the connection is unhealthy", async () => {
      const user = userEvent.setup();
      testConnection.mockResolvedValue({ status: "unhealthy", error: "connection refused" });
      renderSettings();

      await screen.findByLabelText("Host");
      await user.click(screen.getByRole("button", { name: /test connection/i }));

      await waitFor(() =>
        expect(notifications.fromBackend).toHaveBeenCalledWith(expect.stringContaining("connection refused")),
      );
      expect(notifications.success).not.toHaveBeenCalled();
    });
  });
});
