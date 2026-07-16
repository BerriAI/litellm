import { renderWithProviders, screen } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { Form } from "antd";
import AddAutoRouterTab from "./add_auto_router_tab";
import NotificationManager from "../molecules/notifications_manager";

vi.mock("../networking", () => ({
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([{ model_group: "gpt-4o", mode: "chat" }]),
}));

vi.mock("./handle_add_auto_router_submit", () => ({
  handleAddAutoRouterSubmit: vi.fn(),
}));

vi.mock("../molecules/notifications_manager", () => ({
  default: { fromBackend: vi.fn() },
}));

const Harness = () => {
  const [form] = Form.useForm();
  return <AddAutoRouterTab form={form} handleOk={vi.fn()} accessToken="token" userRole="Admin" />;
};

describe("AddAutoRouterTab", () => {
  it("flags every mandatory field when Add Auto Router is clicked with nothing filled", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Harness />);

    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    expect(await screen.findByText("Auto router name is required")).toBeInTheDocument();
    expect(screen.getAllByText("This tier is required")).toHaveLength(4);
    expect(NotificationManager.fromBackend).toHaveBeenCalledWith("Please enter an Auto Router Name");
  });

  it("blocks the semantic router submit until an embedding model is selected", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Harness />);

    await user.click(screen.getByRole("radio", { name: /semantic router/i }));
    await user.type(screen.getByPlaceholderText(/smart_router/i), "sem_router");

    const defaultModelSelect = await screen.findByRole("combobox", { name: /default model/i });
    await user.click(defaultModelSelect);
    await user.click(await screen.findByTitle("gpt-4o"));

    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    expect(NotificationManager.fromBackend).toHaveBeenCalledWith("Please select an Embedding Model");
  });
});
