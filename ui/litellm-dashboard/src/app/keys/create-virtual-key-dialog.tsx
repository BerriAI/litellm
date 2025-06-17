import { cx } from "@/lib/cva.config";
import { DialogStore, useDialogContext, useDialogStore } from "@ariakit/react";
import {
  UiFormCombobox,
  UiFormContent,
  UiFormDescription,
  UiFormError,
  UiFormGroup,
  UiFormLabel,
  UiFormLabelGroup,
  UiFormRadio,
  UiFormRadioGroup,
  UiFormTextInput,
  UiModelSelect,
} from "./forms";
import { UiDialog } from "./ui-dialog";
import { Controller, SubmitHandler, useForm } from "react-hook-form";
import { AuthContext, useAuthContext } from "./contexts";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Fragment, useState } from "react";
import { keyCreateCall } from "@/components/networking";
import { message } from "antd";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { CopyApiKeyDialog } from "./copy-api-key-dialog";
import { Record } from "openai/core.mjs";
import { modelAvailableCall, teamListCall } from "../../components/networking";
import { Team } from "../../components/key_team_helpers/key_list";
import { AiModel } from "../../types";

type CreateVirtualKeyFormData = {
  owner: "you" | "service_account";
  key_alias: string;
  team_id: string;
  modelAccess: "all_team_models" | "select_models";
  models: string[];
};

type CreateVirtualKeyPayload = {
  key_alias: string;
  user_id: string;
  team_id?: string | null;
  models: string[];
  metadata?: string;
};

function transformFormSubmissionData(args: {
  data: CreateVirtualKeyFormData;
  authCtx: AuthContext;
}): CreateVirtualKeyPayload {
  const { data, authCtx } = args;
  const metadata: Record<string, any> = {};

  if (data.owner === "service_account") {
    metadata.service_account_id = data.key_alias;
  }

  return {
    key_alias: data.key_alias,
    user_id: authCtx.user_id,
    team_id: data.team_id ? data.team_id : undefined,
    models:
      data.modelAccess === "all_team_models"
        ? ["all-team-models"]
        : data.models,
    metadata:
      Object.keys(metadata).length > 0 ? JSON.stringify(metadata) : undefined,
  };
}

function CreateVirtualKeyForm() {
  const authCtx = useAuthContext();
  const createVirtualKeyDialogStore = useDialogContext();
  const queryClient = useQueryClient();

  const [copyApiKeyDialogUiState, setCopyApiKeyDialogUiState] = useState<
    { type: "hidden" } | { type: "visible"; apiKey: string }
  >({ type: "hidden" });

  const copyApiKeyDialogStore = useDialogStore({
    setOpen(open) {
      if (open === false) {
        createVirtualKeyDialogStore?.hide();
      }
    },
  });

  const form = useForm<CreateVirtualKeyFormData>({
    defaultValues: {
      owner: "you",
      key_alias: "",
      models: [],
      modelAccess: "select_models",
    },
  });
  const { errors } = form.formState;
  const modelAccess = form.watch("modelAccess");
  const owner = form.watch("owner");

  const createVirtualKeyMutation = useMutation({
    mutationFn: (payload: CreateVirtualKeyPayload) => {
      return keyCreateCall(authCtx.key, authCtx.user_id, payload);
    },
    onSuccess(data: { key: string }) {
      queryClient.invalidateQueries({ queryKey: ["keys"] });
      setCopyApiKeyDialogUiState({ type: "visible", apiKey: data.key });
      copyApiKeyDialogStore.show();
    },
    onError(error) {
      message.error(parseErrorMessage(error));
    },
  });

  const teamsQuery = useQuery<Team[]>({
    initialData: () => [],
    queryKey: ["teams"],
    queryFn: () => teamListCall(authCtx.key, null),
  });

  const modelsQuery = useQuery<AiModel[]>({
    queryKey: ["models"],
    initialData: () => [],
    queryFn: () =>
      modelAvailableCall(authCtx.key, authCtx.user_id, authCtx.user_role).then(
        (res) => res.data,
      ),
  });

  const teams = teamsQuery.data;
  const models = modelsQuery.data;

  const busy = createVirtualKeyMutation.isPending;

  const onSubmit: SubmitHandler<CreateVirtualKeyFormData> = (data) => {
    createVirtualKeyMutation.mutate(
      transformFormSubmissionData({
        data,
        authCtx: authCtx,
      }),
    );
  };

  return (
    <Fragment>
      {copyApiKeyDialogUiState.type === "visible" ? (
        <CopyApiKeyDialog
          store={copyApiKeyDialogStore}
          apiKey={copyApiKeyDialogUiState.apiKey}
        />
      ) : null}

      <form
        onSubmit={form.handleSubmit(onSubmit)}
        className="flex flex-col h-full min-h-0"
      >
        <div className="py-[56px] px-[48px] block min-h-0 overflow-y-auto">
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
                <Controller
                  control={form.control}
                  name="owner"
                  render={({ field }) => (
                    <UiFormRadioGroup
                      value={field.value}
                      setValue={field.onChange}
                    >
                      <UiFormRadio value="you">You</UiFormRadio>
                      <UiFormRadio value="service_account">
                        Service Account
                      </UiFormRadio>
                    </UiFormRadioGroup>
                  )}
                />
              </UiFormContent>
            </UiFormGroup>

            <UiFormGroup>
              <UiFormLabelGroup>
                <UiFormLabel>
                  {owner === "service_account"
                    ? "Service Account ID"
                    : "Key name"}
                </UiFormLabel>

                <UiFormDescription>
                  {owner === "service_account"
                    ? "A descriptive name to identify this service account and distinguish it from other service accounts"
                    : "A descriptive name to identify this key and distinguish it from other virtual keys"}
                </UiFormDescription>
              </UiFormLabelGroup>

              <UiFormContent>
                <UiFormTextInput
                  {...form.register("key_alias", {
                    validate(value) {
                      return (
                        value.trim().length > 0 ||
                        (owner === "service_account"
                          ? "Service Account ID cannot be empty"
                          : "Key name cannot be empty")
                      );
                    },
                  })}
                  placeholder={
                    owner === "service_account"
                      ? "Service Account ID"
                      : "Name of Key"
                  }
                />

                {errors.key_alias ? (
                  <UiFormError>{errors.key_alias.message}</UiFormError>
                ) : null}
              </UiFormContent>
            </UiFormGroup>

            <UiFormGroup>
              <UiFormLabelGroup>
                <UiFormLabel>Team</UiFormLabel>
                <UiFormDescription>
                  The team this key belongs to, which determines available
                  models, budget limits, and access permissions. Team assignment
                  controls resource allocation and usage tracking for this
                  virtual key
                </UiFormDescription>
              </UiFormLabelGroup>

              <UiFormContent>
                <Controller
                  control={form.control}
                  name="team_id"
                  render={({ field }) => (
                    <UiFormCombobox
                      placeholder="Select Team"
                      value={field.value}
                      setValue={field.onChange}
                      items={teams.map((team) => ({
                        title: team.team_alias,
                        subtitle: team.team_id,
                        value: team.team_id,
                      }))}
                    />
                  )}
                />
              </UiFormContent>
            </UiFormGroup>

            <UiFormGroup>
              <UiFormLabelGroup>
                <UiFormLabel>Models</UiFormLabel>
                <UiFormDescription>
                  Select which models this key can access. Choose All Team
                  Models to grant access to all models available to the team, or
                  pick individual models to create custom access permissions for
                  this key
                </UiFormDescription>
              </UiFormLabelGroup>

              <UiFormContent>
                <Controller
                  control={form.control}
                  name="modelAccess"
                  render={({ field }) => (
                    <UiFormRadioGroup
                      value={field.value}
                      setValue={field.onChange}
                    >
                      <UiFormRadio value="all_team_models">
                        All team models
                      </UiFormRadio>
                      <UiFormRadio value="select_models">
                        Select specific models
                      </UiFormRadio>
                    </UiFormRadioGroup>
                  )}
                />

                {modelAccess === "select_models" ? (
                  <Fragment>
                    <Controller
                      control={form.control}
                      name="models"
                      rules={{
                        validate(value) {
                          return value.length > 0 || "Please select models";
                        },
                      }}
                      render={({ field }) => (
                        <UiModelSelect
                          value={field.value}
                          setValue={field.onChange}
                          models={models}
                          ref={field.ref}
                        />
                      )}
                    />

                    {errors.models ? (
                      <UiFormError>{errors.models.message}</UiFormError>
                    ) : null}
                  </Fragment>
                ) : null}
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
            onClick={createVirtualKeyDialogStore?.hide}
            type="button"
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
            disabled={busy}
            type="submit"
            className={cx(
              "h-[36px] px-3 rounded-md",
              "flex items-center gap-1",
              "text-[12px] font-medium tracking-tight",
              "indigo-button-3d disabled:opacity-50",
            )}
          >
            <span className="text-white">Create Key</span>
            {busy ? (
              <UiLoadingSpinner className="size-3 ml-1 text-white" />
            ) : null}
          </button>
        </div>
      </form>
    </Fragment>
  );
}

type CreateVirtualKeyDialogProps = {
  store?: DialogStore;
};

export function CreateVirtualKeyDialog(props: CreateVirtualKeyDialogProps) {
  return (
    <UiDialog store={props.store}>
      <CreateVirtualKeyForm />
    </UiDialog>
  );
}
