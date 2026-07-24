import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  createGuardrailCall,
  deleteGuardrailCall,
  getGuardrailsList,
  updateGuardrailCall,
} from "@/components/networking";

vi.mock("@/components/networking", () => ({
  getGuardrailsList: vi.fn(),
  createGuardrailCall: vi.fn(),
  updateGuardrailCall: vi.fn(),
  deleteGuardrailCall: vi.fn(),
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  __esModule: true,
  default: { success: vi.fn(), fromBackend: vi.fn(), error: vi.fn() },
}));

import PromptCompressionTab from "./PromptCompressionTab";

const headroomGuardrail = {
  guardrail_id: "gr-1",
  guardrail_name: "headroom-compression",
  litellm_params: { guardrail: "headroom", api_base: "https://headroom.example.com", default_on: true },
};

const mockedList = vi.mocked(getGuardrailsList);
const mockedCreate = vi.mocked(createGuardrailCall);
const mockedUpdate = vi.mocked(updateGuardrailCall);
const mockedDelete = vi.mocked(deleteGuardrailCall);

const masterSwitch = () => screen.getByLabelText("Toggle Headroom compression");

beforeEach(() => {
  vi.clearAllMocks();
});

describe("PromptCompressionTab - no guardrail configured", () => {
  beforeEach(() => {
    mockedList.mockResolvedValue({ guardrails: [] });
  });

  it("renders a single permanent card with no separate add-guardrail section", async () => {
    render(<PromptCompressionTab accessToken="sk-test" />);

    await waitFor(() => expect(screen.getByText("Disabled")).toBeInTheDocument());
    expect(screen.getByText("Headroom prompt compression")).toBeInTheDocument();
    expect(screen.queryByText("Add Headroom compression guardrail")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Add guardrail" })).not.toBeInTheDocument();
    expect(masterSwitch()).not.toBeChecked();
  });

  it("does not create a guardrail when enabling with empty required fields", async () => {
    render(<PromptCompressionTab accessToken="sk-test" />);
    await waitFor(() => expect(masterSwitch()).toBeEnabled());

    fireEvent.click(masterSwitch());

    await waitFor(() => expect(screen.getByText("Name is required")).toBeInTheDocument());
    expect(mockedCreate).not.toHaveBeenCalled();
  });

  it("creates a headroom guardrail with the entered name, api base and apply-to-all", async () => {
    mockedCreate.mockResolvedValue({});
    render(<PromptCompressionTab accessToken="sk-test" />);
    await waitFor(() => expect(masterSwitch()).toBeEnabled());

    fireEvent.change(screen.getByPlaceholderText("headroom-compression"), { target: { value: "my-headroom" } });
    fireEvent.change(screen.getByPlaceholderText("https://your-headroom-endpoint"), {
      target: { value: "https://compress.internal" },
    });
    fireEvent.click(masterSwitch());

    await waitFor(() => expect(mockedCreate).toHaveBeenCalledTimes(1));
    expect(mockedCreate).toHaveBeenCalledWith("sk-test", {
      guardrail_name: "my-headroom",
      litellm_params: {
        guardrail: "headroom",
        mode: "pre_call",
        api_base: "https://compress.internal",
        default_on: true,
      },
    });
  });
});

describe("PromptCompressionTab - guardrail already configured", () => {
  beforeEach(() => {
    mockedList.mockResolvedValue({ guardrails: [headroomGuardrail] });
  });

  it("shows the enabled card populated from the existing guardrail", async () => {
    render(<PromptCompressionTab accessToken="sk-test" />);

    await waitFor(() => expect(screen.getByText("Enabled")).toBeInTheDocument());
    expect(masterSwitch()).toBeChecked();
    expect(screen.getByPlaceholderText("headroom-compression")).toHaveValue("headroom-compression");
    expect(screen.getByPlaceholderText("https://your-headroom-endpoint")).toHaveValue("https://headroom.example.com");
  });

  it("keeps Save changes disabled until a field is edited, then patches the guardrail", async () => {
    mockedUpdate.mockResolvedValue({});
    render(<PromptCompressionTab accessToken="sk-test" />);
    await waitFor(() => expect(screen.getByText("Enabled")).toBeInTheDocument());

    const save = screen.getByRole("button", { name: "Save changes" });
    expect(save).toBeDisabled();

    fireEvent.change(screen.getByPlaceholderText("https://your-headroom-endpoint"), {
      target: { value: "https://new-endpoint.internal" },
    });
    await waitFor(() => expect(save).toBeEnabled());

    fireEvent.click(save);
    await waitFor(() => expect(mockedUpdate).toHaveBeenCalledTimes(1));
    expect(mockedUpdate).toHaveBeenCalledWith("sk-test", "gr-1", {
      guardrail_name: "headroom-compression",
      litellm_params: {
        guardrail: "headroom",
        mode: "pre_call",
        api_base: "https://new-endpoint.internal",
        default_on: true,
      },
    });
  });

  it("deletes the guardrail after confirming the turn-off prompt", async () => {
    mockedDelete.mockResolvedValue({});
    render(<PromptCompressionTab accessToken="sk-test" />);
    await waitFor(() => expect(screen.getByText("Enabled")).toBeInTheDocument());

    fireEvent.click(masterSwitch());

    const confirm = await screen.findByRole("button", { name: "Turn off" });
    fireEvent.click(confirm);

    await waitFor(() => expect(mockedDelete).toHaveBeenCalledWith("sk-test", "gr-1"));
    await waitFor(() => expect(screen.getByText("Disabled")).toBeInTheDocument());
    expect(mockedUpdate).not.toHaveBeenCalled();
    expect(mockedCreate).not.toHaveBeenCalled();
  });
});
