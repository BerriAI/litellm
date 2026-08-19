// eslint-disable-next-line no-restricted-imports -- the parent cost_tracking_settings still owns this antd Form, and the point of this test is that AddProviderForm registers nothing in it
import { Form } from "antd";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../../../tests/test-utils";
import AddProviderForm from "./add_provider_form";
import { DiscountConfig } from "./types";

const onAddProvider = vi.fn();
const onParentFinish = vi.fn();
const readParentStore = vi.fn();

const ParentOwnedForm = () => {
  const [form] = Form.useForm();
  return (
    <Form form={form} onFinish={onParentFinish} layout="vertical" className="space-y-6">
      <AddProviderForm
        discountConfig={{} as DiscountConfig}
        selectedProvider="OpenAI"
        newDiscount="5"
        onProviderChange={vi.fn()}
        onDiscountChange={vi.fn()}
        onAddProvider={() => {
          readParentStore(form.getFieldsValue());
          onAddProvider();
        }}
      />
    </Form>
  );
};

describe("AddProviderForm inside the antd form its parent owns", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("registers no field in the parent FormInstance, so the parent's resetFields is a no-op", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ParentOwnedForm />);

    await user.click(screen.getByRole("button", { name: /add provider discount/i }));

    expect(readParentStore).toHaveBeenCalledTimes(1);
    expect(readParentStore.mock.calls[0][0]).toEqual({});
  });

  it("drives both the onAddProvider prop and the parent form submit from one click", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ParentOwnedForm />);

    await user.click(screen.getByRole("button", { name: /add provider discount/i }));

    expect(onAddProvider).toHaveBeenCalledTimes(1);
    expect(onParentFinish).toHaveBeenCalledTimes(1);
  });

  it("treats Enter in the discount field exactly like a click on the add button", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ParentOwnedForm />);

    await user.type(screen.getByPlaceholderText("5"), "{Enter}");

    await vi.waitFor(() => expect(onParentFinish).toHaveBeenCalledTimes(1));
    expect(onAddProvider).toHaveBeenCalledTimes(1);
  });
});
