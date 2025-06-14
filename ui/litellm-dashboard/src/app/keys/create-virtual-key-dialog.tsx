import { cx } from "@/lib/cva.config";
import {
  Dialog,
  DialogStore,
  useDialogContext,
  useDialogStore,
} from "@ariakit/react";
import {
  UiFormCombobox,
  UiFormContent,
  UiFormDescription,
  UiFormGroup,
  UiFormLabel,
  UiFormLabelGroup,
  UiFormRadio,
  UiFormRadioGroup,
  UiFormTextInput,
  UiModelSelect,
} from "./forms";
import { teams } from "./data";

function Content() {
  const store = useDialogContext();

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="py-[56px] px-[48px] min-h-0 overflow-y-auto">
        <div className="flex flex-col gap-1 mb-[48px]">
          <h1 className="text-[28px] tracking-tighter text-neutral-900">
            Create a new Virtual Key
          </h1>

          <p className="max-w-[480px] text-[15px]/[1.6] text-neutral-500 tracking-tight">
            Set up a new virtual key with{" "}
            <span className="font-medium text-neutral-900">configurable</span>{" "}
            settings for{" "}
            <span className="font-medium text-neutral-900">secure</span> API
            access management
          </p>
        </div>

        <div
          className={cx(
            "bg-white rounded-lg",
            "ring-[0.5px] ring-black/[0.08]",
            "px-6 py-2",
          )}
        >
          <UiFormGroup>
            <UiFormLabelGroup>
              <UiFormLabel>Owner of this Key</UiFormLabel>
              <UiFormDescription>
                Choose You for individual ownership, or Service Account for
                organizational ownership that allows multiple users to access
                the key
              </UiFormDescription>
            </UiFormLabelGroup>

            <UiFormContent>
              <UiFormRadioGroup defaultValue="you">
                <UiFormRadio value="you">You</UiFormRadio>
                <UiFormRadio value="service_account">
                  Service Account
                </UiFormRadio>
              </UiFormRadioGroup>
            </UiFormContent>
          </UiFormGroup>

          <UiFormGroup>
            <UiFormLabelGroup>
              <UiFormLabel>Key name</UiFormLabel>
              <UiFormDescription>
                A descriptive name to identify this key and distinguish it from
                other virtual keys
              </UiFormDescription>
            </UiFormLabelGroup>

            <UiFormContent>
              <UiFormTextInput placeholder="Name of Key" />
            </UiFormContent>
          </UiFormGroup>

          <UiFormGroup>
            <UiFormLabelGroup>
              <UiFormLabel>Team</UiFormLabel>
              <UiFormDescription>
                The team this key belongs to, which determines available models,
                budget limits, and access permissions. Team assignment controls
                resource allocation and usage tracking for this virtual key
              </UiFormDescription>
            </UiFormLabelGroup>

            <UiFormContent>
              <UiFormCombobox
                placeholder="Select Team"
                items={teams.map((team) => ({
                  title: team.name,
                  subtitle: team.id,
                }))}
              />
            </UiFormContent>
          </UiFormGroup>

          <UiFormGroup>
            <UiFormLabelGroup>
              <UiFormLabel>Models</UiFormLabel>
              <UiFormDescription>
                Select which models this key can access. Choose All Team Models
                to grant access to all models available to the team, or pick
                individual models to create custom access permissions for this
                key
              </UiFormDescription>
            </UiFormLabelGroup>

            <UiFormContent>
              <UiFormRadioGroup defaultValue="select_models">
                <UiFormRadio value="all_team_models">
                  All team models
                </UiFormRadio>
                <UiFormRadio value="select_models">
                  Select specific models
                </UiFormRadio>
              </UiFormRadioGroup>

              <UiModelSelect />
            </UiFormContent>
          </UiFormGroup>
        </div>
      </div>

      <div
        className={cx(
          "flex items-center justify-between mt-auto bg-neutral-100",
          "px-8 h-[64px] shrink-0",
        )}
      >
        <button
          onClick={store?.hide}
          className={cx(
            "h-[36px] bg-white px-3 rounded-md",
            "flex items-center gap-1",
            "text-[12px] font-medium tracking-tight",
            "bg-white",
          )}
        >
          <span className="text-neutral-900">Cancel</span>
        </button>

        <button
          type="button"
          className={cx(
            "h-[36px] px-3 rounded-md",
            "flex items-center gap-1",
            "text-[12px] font-medium tracking-tight",
            "indigo-button-3d",
          )}
        >
          <span className="text-white">Create Key</span>
        </button>
      </div>
    </div>
  );
}

type CreateVirtualKeyDialogProps = {
  store?: DialogStore;
};

export function CreateVirtualKeyDialog(props: CreateVirtualKeyDialogProps) {
  const store = useDialogStore({ store: props.store });

  return (
    <Dialog
      autoFocusOnShow={false}
      store={store}
      backdrop={
        <div
          className={cx(
            "fixed inset-0 bg-black/20",
            "transition-[opacity,backdrop-filter] duration-200 ease-out",
            "backdrop-blur-0 opacity-0",
            "data-[enter]:opacity-100 data-[enter]:backdrop-blur-sm",
          )}
        />
      }
      unmountOnHide
      className={cx(
        "inset-12 m-auto fixed bg-[#FBFBFB]",
        "outline-none rounded-2xl overflow-x-hidden",
        "ring-[0.5px] ring-black/[0.08]",
        "shadow-2xl",
        "max-w-[1020px] max-h-[min(90%,1200px)]",
        "origin-center transition-[opacity,transform] duration-200 ease-out",
        "scale-95 opacity-0",
        "data-[enter]:scale-100 data-[enter]:opacity-100",
      )}
    >
      <Content />
    </Dialog>
  );
}
