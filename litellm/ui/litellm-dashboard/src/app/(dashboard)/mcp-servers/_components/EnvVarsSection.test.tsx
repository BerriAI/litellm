import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FormProvider, useForm } from "react-hook-form";

import {
  MountedFormProvider,
  projectMountedValues,
  useMountRegistry,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";
import EnvVarsSection from "./EnvVarsSection";

const renderSection = (defaultValues: MountedFormValues) => {
  const onFinish = vi.fn();
  const Harness: React.FC = () => {
    const form = useForm<MountedFormValues>({ mode: "onChange", defaultValues });
    const registry = useMountRegistry();
    return (
      <FormProvider {...form}>
        <MountedFormProvider value={{ control: form.control, registry }}>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              onFinish(projectMountedValues(registry, form.getValues));
            }}
          >
            <EnvVarsSection />
            <button type="submit">Submit</button>
          </form>
        </MountedFormProvider>
      </FormProvider>
    );
  };
  render(<Harness />);
  return onFinish;
};

describe("EnvVarsSection", () => {
  it("submits a per-user row whole, keeping the value key whose input the scope hides", async () => {
    const onFinish = renderSection({
      env_vars: [{ name: "DB_USER", value: "admin", scope: "user", description: "Your DB username" }],
    });

    expect(screen.queryByPlaceholderText("e.g. postgresql")).not.toBeInTheDocument();
    await userEvent.click(screen.getByText("Submit"));

    expect(onFinish).toHaveBeenCalledWith(
      expect.objectContaining({
        env_vars: [{ name: "DB_USER", value: "admin", scope: "user", description: "Your DB username" }],
      }),
    );
  });

  it("submits an empty env_vars key when the list has no rows, rather than dropping the key", async () => {
    const onFinish = renderSection({ env_vars: [] });

    await userEvent.click(screen.getByText("Submit"));

    expect(onFinish.mock.calls[0][0]).toHaveProperty("env_vars", []);
  });

  it("carries a row added after mount into the submitted list, scoped global without the user picking one", async () => {
    const onFinish = renderSection({ env_vars: [] });

    await userEvent.click(screen.getByText("Add Variable"));
    await userEvent.type(screen.getByPlaceholderText("e.g. DB_PROTOCOL"), "DB_PROTOCOL");
    await userEvent.type(screen.getByPlaceholderText("e.g. postgresql"), "postgresql");
    await userEvent.click(screen.getByText("Submit"));

    expect(onFinish).toHaveBeenCalledWith(
      expect.objectContaining({
        env_vars: [expect.objectContaining({ name: "DB_PROTOCOL", value: "postgresql", scope: "global" })],
      }),
    );
  });

  it("rejects a variable name that starts with a digit", async () => {
    renderSection({ env_vars: [{ name: "", value: "", scope: "global", description: "" }] });

    await userEvent.type(screen.getByPlaceholderText("e.g. DB_PROTOCOL"), "9LIVES");

    expect(await screen.findByText("Use letters, digits, underscores; cannot start with a digit.")).toBeInTheDocument();
  });
});
