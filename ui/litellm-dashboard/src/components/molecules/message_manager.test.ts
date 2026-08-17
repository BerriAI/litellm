import { createElement } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const sonner = vi.hoisted(() => ({
  success: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  error: vi.fn(),
  dismiss: vi.fn(),
}));

vi.mock("sonner", () => ({ toast: sonner }));
vi.mock("@/components/molecules/notifications_manager", () =>
  vi.importActual<typeof import("./notifications_manager")>("./notifications_manager"),
);

import MessageManager from "./message_manager";
import NotificationManager from "./notifications_manager";

describe("legacy toast facades", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("MessageManager converts antd-era seconds into milliseconds", () => {
    MessageManager.error("failed!", 5);
    expect(sonner.error).toHaveBeenCalledWith("failed!", { description: undefined, duration: 5000 });
  });

  it("MessageManager falls back to the kind default when no duration is given", () => {
    MessageManager.success("done!");
    expect(sonner.success).toHaveBeenCalledWith("done!", { description: undefined, duration: 4000 });
  });

  it("MessageManager.destroy and NotificationManager.clear both dismiss", () => {
    MessageManager.destroy();
    NotificationManager.clear();
    expect(sonner.dismiss).toHaveBeenCalledTimes(2);
  });

  it("NotificationManager.fromBackend routes through toast.fromError", () => {
    NotificationManager.fromBackend({ message: "Team not found", type: "not_found_error", code: "404" });
    expect(sonner.error).toHaveBeenCalledWith("Not Found", { description: "Team not found", duration: 6000 });
  });

  it("NotificationManager.fromBackend converts the antd-era extra.duration seconds", () => {
    NotificationManager.fromBackend("boom", { duration: 8 });
    expect(sonner.error).toHaveBeenCalledWith("Error", { description: "boom", duration: 8000 });
  });

  it("NotificationManager keeps the antd config-object form: message is the title, description below it", () => {
    NotificationManager.success({
      message: "MCP Server submitted for admin review",
      description: "Once an admin approves it, the server will appear in your MCP Servers list.",
      duration: 10,
    });
    expect(sonner.success).toHaveBeenCalledWith("MCP Server submitted for admin review", {
      description: "Once an admin approves it, the server will appear in your MCP Servers list.",
      duration: 10000,
    });
  });

  it("NotificationManager falls back to the kind's title when a config object has no message", () => {
    NotificationManager.warning({ description: "Heads up" });
    expect(sonner.warning).toHaveBeenCalledWith("Warning", { description: "Heads up", duration: 6000 });
  });

  it("NotificationManager treats a React element as the message, not a config object", () => {
    const element = createElement("span", null, "done");
    NotificationManager.info(element);
    expect(sonner.info).toHaveBeenCalledWith(element, { description: undefined, duration: 4000 });
  });
});
