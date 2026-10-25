"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { BotIcon, InfoIcon, LayersIcon, ServerIcon, SparklesIcon } from "lucide-react";
import * as React from "react";

import { accessGroupKeys } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";
import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { ModelSelect } from "@/components/ModelSelect/ModelSelect";
import { toast } from "@/lib/toast";
import TranslatedMessage from "@/components/shared/TranslatedMessage";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { MultiSelect } from "@/components/shared/MultiSelect";
import SkillSelector from "@/components/skills/SkillSelector";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { useZodForm } from "@/lib/forms/useZodForm";
import { fetchClient } from "@/lib/http/api";

import { AccessGroupMCPToolPermissions } from "../AccessGroupsModal/AccessGroupBaseForm";
import { buildAccessGroupCreateBody, emptyAccessGroupFormValues, type AccessGroupCreateBody } from "./mapper";
import { accessGroupCreateSchema } from "./schema";

const GENERAL_TAB = "general";

const defaultCreateAccessGroup = async (body: AccessGroupCreateBody): Promise<unknown> => {
  const { data } = await fetchClient.POST("/v1/access_group", { body });
  return data;
};

interface AccessGroupCreateDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  createAccessGroup?: (body: AccessGroupCreateBody) => Promise<unknown>;
}

export const AccessGroupCreateDialog = ({
  open,
  onOpenChange,
  createAccessGroup = defaultCreateAccessGroup,
}: AccessGroupCreateDialogProps) => {
  const { t } = useLanguage();
  const queryClient = useQueryClient();
  // 名称校验保留固定原文，切换语言只更新字段错误的显示。
  const form = useZodForm(accessGroupCreateSchema, { defaultValues: emptyAccessGroupFormValues });
  const [activeTab, setActiveTab] = React.useState(GENERAL_TAB);

  const { accessToken } = useAuthorized();
  const { data: agentsData } = useAgents();
  const { data: mcpServersData } = useMCPServers();

  const mcpServerOptions = (mcpServersData ?? []).map((server) => ({
    value: server.server_id,
    label: server.server_name ?? server.server_id,
  }));
  const agentOptions = (agentsData?.agents ?? []).map((agent) => ({
    value: agent.agent_id,
    label: agent.agent_name,
  }));

  const closeAndReset = () => {
    form.reset(emptyAccessGroupFormValues);
    setActiveTab(GENERAL_TAB);
    onOpenChange(false);
  };

  const mutation = useMutation({
    mutationFn: (body: AccessGroupCreateBody) => createAccessGroup(body),
    onSuccess: () => {
      // 固定通知按当前语言渲染，不把服务端错误消息作为翻译键。
      toast.success(<TranslatedMessage message="Access group created successfully" />);
      queryClient.invalidateQueries({ queryKey: accessGroupKeys.all });
      closeAndReset();
    },
    onError: (error: unknown) =>
      toast.fromError(
        error,
        error instanceof Error && error.message
          ? undefined
          : { description: <TranslatedMessage message="Failed to create access group" /> },
      ),
  });

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen && mutation.isPending) return;
    if (!nextOpen) {
      form.reset(emptyAccessGroupFormValues);
      setActiveTab(GENERAL_TAB);
    }
    onOpenChange(nextOpen);
  };

  const onSubmit = form.handleSubmit(
    (values) => {
      if (mutation.isPending) return;
      mutation.mutate(buildAccessGroupCreateBody(values));
    },
    // the only validated field (name) lives on the General Info tab
    () => setActiveTab(GENERAL_TAB),
  );

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{t("Create Access Group")}</DialogTitle>
        </DialogHeader>

        <form onSubmit={onSubmit} noValidate>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="w-full">
              <TabsTrigger value={GENERAL_TAB}>
                <InfoIcon />
                {t("General Info")}
              </TabsTrigger>
              <TabsTrigger value="models">
                <LayersIcon />
                {t("Models")}
              </TabsTrigger>
              <TabsTrigger value="mcp-servers">
                <ServerIcon />
                {t("MCP Servers")}
              </TabsTrigger>
              <TabsTrigger value="agents">
                <BotIcon />
                {t("Agents")}
              </TabsTrigger>
              <TabsTrigger value="skills">
                <SparklesIcon />
                {t("Skills")}
              </TabsTrigger>
            </TabsList>

            <TabsContent value={GENERAL_TAB} className="pt-4">
              <FieldGroup>
                <FormField
                  control={form.control}
                  name="name"
                  label={t("Group Name")}
                  renderError={(error) => t(error.message ?? "")}
                >
                  {({ ref, ...field }) => <Input {...field} ref={ref} placeholder={t("e.g. Engineering Team")} />}
                </FormField>
                <FormField control={form.control} name="description" label={t("Description")}>
                  {({ ref, ...field }) => (
                    <Textarea
                      {...field}
                      ref={ref}
                      rows={4}
                      placeholder={t("Describe the purpose of this access group...")}
                    />
                  )}
                </FormField>
              </FieldGroup>
            </TabsContent>

            <TabsContent value="models" className="pt-4">
              <FormField control={form.control} name="modelIds" label={t("Allowed Models")}>
                {(field) => <ModelSelect context="global" value={field.value} onChange={field.onChange} />}
              </FormField>
            </TabsContent>

            <TabsContent value="mcp-servers" className="pt-4">
              <FormField control={form.control} name="mcpServerIds" label={t("Allowed MCP Servers")}>
                {({ id, value, onChange }) => (
                  <MultiSelect
                    id={id}
                    value={value}
                    onValueChange={onChange}
                    options={mcpServerOptions}
                    placeholder={t("Select MCP servers")}
                  />
                )}
              </FormField>
              <AccessGroupMCPToolPermissions
                selectedServers={form.watch("mcpServerIds")}
                toolPermissions={form.watch("mcpToolPermissions")}
                onChange={(next) => form.setValue("mcpToolPermissions", next, { shouldDirty: true })}
              />
            </TabsContent>

            <TabsContent value="agents" className="pt-4">
              <FormField control={form.control} name="agentIds" label={t("Allowed Agents")}>
                {({ id, value, onChange }) => (
                  <MultiSelect
                    id={id}
                    value={value}
                    onValueChange={onChange}
                    options={agentOptions}
                    placeholder={t("Select agents")}
                  />
                )}
              </FormField>
            </TabsContent>

            <TabsContent value="skills" className="pt-4">
              <FormField
                control={form.control}
                name="skillNames"
                label={t("Allowed Skills")}
                description={t(
                  "Enabled skills are visible to everyone. Grant disabled (private) Claude Code plugins to the teams and keys in this group here.",
                )}
              >
                {({ value, onChange }) => (
                  <SkillSelector
                    onChange={onChange}
                    value={value}
                    accessToken={accessToken || ""}
                    placeholder={t("Select skills (optional)")}
                  />
                )}
              </FormField>
            </TabsContent>
          </Tabs>

          <DialogFooter className="mt-6">
            <Button
              type="button"
              variant="outline"
              onClick={() => handleOpenChange(false)}
              disabled={mutation.isPending}
            >
              {t("Cancel")}
            </Button>
            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? t("Creating...") : t("Create Group")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};
