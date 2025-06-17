import { cx } from "@/lib/cva.config";
import { Button, DialogStore, useDialogContext } from "@ariakit/react";
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
import {
  Controller,
  DefaultValues,
  SubmitHandler,
  useForm,
} from "react-hook-form";
import { useAuthContext, useGlobalOverlaysContext } from "./contexts";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Fragment, useState } from "react";
import { keyUpdateCall } from "@/components/networking";
import { message } from "antd";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { Record } from "openai/core.mjs";
import { modelAvailableCall, teamListCall } from "../networking";
import { KeyResponse, Team } from "../key_team_helpers/key_list";
import { AiModel } from "../../types";
import { updateExistingKeys } from "@/utils/dataUtils";
import { Trash2Icon } from "lucide-react";

type EditVirtualKeyFormData = {
  owner: "you" | "service_account";
  key_alias: string;
  team_id: string;
  modelAccess: "all_team_models" | "select_models";
  models: string[];
};

type EditVirtualKeyPayload = {
  key: string;
  key_alias: string;
  team_id?: string | null;
  models: string[];
  metadata?: Record<string, any>;
};

function transformFormSubmissionData(args: {
  data: EditVirtualKeyFormData;
  virtualKey: KeyResponse;
}): EditVirtualKeyPayload {
  const { data, virtualKey } = args;
  const metadata: Record<string, any> = virtualKey.metadata || {};

  if (data.owner === "service_account") {
    metadata.service_account_id = data.key_alias;
  }

  return {
    key: virtualKey.token,
    key_alias: data.key_alias,
    team_id: data.team_id ? data.team_id : null,
    models:
      data.modelAccess === "all_team_models"
        ? ["all-team-models"]
        : data.models,
    metadata,
  };
}

type ContentProps = {
  virtualKey: KeyResponse;
};

function Content(props: ContentProps) {
  const authCtx = useAuthContext();
  const editVirtualKeyDialogStore = useDialogContext();
  const queryClient = useQueryClient();
  const virtualKey = props.virtualKey;
  const overlays = useGlobalOverlaysContext();

  const [defaultValues] = useState<DefaultValues<EditVirtualKeyFormData>>(
    () => {
      const metadata: Record<string, any> = virtualKey.metadata || {};
      const modelAccess = virtualKey.models.includes("all-team-models")
        ? "all_team_models"
        : "select_models";

      const _defaultValues: DefaultValues<EditVirtualKeyFormData> = {
        key_alias: virtualKey.key_alias,
        owner:
          typeof metadata.service_account_id === "string"
            ? "service_account"
            : "you",
        team_id: virtualKey.team_id || "",
        modelAccess,
        models: modelAccess === "all_team_models" ? [] : virtualKey.models,
      };

      return _defaultValues;
    },
  );

  const form = useForm<EditVirtualKeyFormData>({ defaultValues });
  const { errors } = form.formState;
  const modelAccess = form.watch("modelAccess");
  const owner = form.watch("owner");
  const keyAlias = form.watch("key_alias");

  const editVirtualKeyMutation = useMutation({
    mutationFn: (payload: EditVirtualKeyPayload) => {
      return keyUpdateCall(authCtx.key, payload);
    },
    onSuccess(updatedVirtualKey: KeyResponse) {
      // Update local cache
      queryClient.setQueriesData(
        { queryKey: ["keys"] },
        (oldData: KeyResponse[] | undefined) => {
          if (oldData === undefined) return undefined;
          return oldData.map((oldVirtualKey) => {
            if (oldVirtualKey.token === virtualKey.token) {
              return updateExistingKeys(oldVirtualKey, updatedVirtualKey);
            }
            return oldVirtualKey;
          });
        },
      );

      queryClient.invalidateQueries({ queryKey: ["keys"] });
      editVirtualKeyDialogStore?.hide();
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

  const busy = editVirtualKeyMutation.isPending;

  const onSubmit: SubmitHandler<EditVirtualKeyFormData> = (data) => {
    editVirtualKeyMutation.mutate(
      transformFormSubmissionData({ data, virtualKey }),
    );
  };

  return (
    <Fragment>
      <form
        onSubmit={form.handleSubmit(onSubmit)}
        className="flex flex-col h-full min-h-0"
      >
        <div className="py-[56px] px-[48px] block min-h-0 overflow-y-auto">
          <div className="flex flex-col gap-4 mb-[48px]">
            <div
              className={cx(
                "flex items-start gap-4 justify-between",
                "border-b border-dashed pb-4",
              )}
            >
              <div className="grow">
                <h1 className="text-[28px] mb-1 tracking-tighter text-neutral-900">
                  Manage Virtual Key
                </h1>

                <p className="max-w-[480px] text-[15px]/[1.6] text-neutral-500 tracking-tight">
                  Modify your virtual key‘s configuration and access controls to
                  meet changing requirements
                </p>
              </div>

              <div className="shrink-0">
                <Button
                  onClick={() =>
                    overlays.deleteVirtualKey({
                      virtualKey,
                      onSuccess() {
                        editVirtualKeyDialogStore?.hide();
                      },
                    })
                  }
                  type="button"
                  className={cx(
                    "h-[32px] bg-white px-3 rounded-md",
                    "flex items-center gap-1",
                    "text-[11px] font-medium text-red-600 tracking-tight",
                    "ring-[0.7px] ring-black/[0.08]",
                    "shadow-md shadow-black/[0.08]",
                  )}
                >
                  <Trash2Icon size={14} />
                  <span>Delete</span>
                </Button>
              </div>
            </div>

            <div className="min-w-0">
              <p className="text-blue-500 mb-2 tracking-tight text-[18px] truncate">
                {keyAlias}
              </p>
              <p className="text-[13px] font-mono text-neutral-800 mb-0.5 truncate">
                {virtualKey.token}
              </p>
              <p className="text-[13px] font-mono text-neutral-500 truncate">
                {virtualKey.key_name}
              </p>
            </div>
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
          <Button
            autoFocus
            onClick={editVirtualKeyDialogStore?.hide}
            type="button"
            className={cx(
              "h-[36px] bg-white px-3 rounded-md",
              "flex items-center gap-1",
              "text-[12px] font-medium tracking-tight",
              "bg-white",
            )}
          >
            <span className="text-neutral-900">Cancel</span>
          </Button>

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
            <span className="text-white">Save changes</span>
            {busy ? (
              <UiLoadingSpinner className="size-3 ml-1 text-white" />
            ) : null}
          </button>
        </div>
      </form>
    </Fragment>
  );
}

export type EditVirtualKeyDialogProps = {
  store?: DialogStore;
  virtualKey: KeyResponse;
};

export function EditVirtualKeyDialog(props: EditVirtualKeyDialogProps) {
  return (
    <UiDialog store={props.store}>
      <Content virtualKey={props.virtualKey} />
    </UiDialog>
  );
}
