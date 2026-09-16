import React from "react";
import { useFormContext } from "react-hook-form";
import { Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { listSapDeploymentsCall, type SapDeploymentInfo } from "../networking";
import { MountedFormField, type MountedFormValues } from "../common_components/MountedFormField";
import { buildSapDeploymentSelection } from "./sapDeploymentSelection";

interface SapDeploymentDiscoveryProps {
  accessToken: string;
}

type DiscoveryState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "loaded"; deployments: SapDeploymentInfo[] };

const SapDeploymentDiscovery: React.FC<SapDeploymentDiscoveryProps> = ({ accessToken }) => {
  const form = useFormContext<MountedFormValues>();
  const [state, setState] = React.useState<DiscoveryState>({ status: "idle" });
  const [selectedId, setSelectedId] = React.useState<string | null>(null);

  const discover = async () => {
    const serviceKey = ((form.getValues("api_key") as string | undefined) ?? "").trim();
    if (!serviceKey) {
      setState({ status: "error", message: "Enter the SAP AI Core service key above first." });
      return;
    }
    const resourceGroup = ((form.getValues("resource_group") as string | undefined) ?? "").trim() || undefined;
    setState({ status: "loading" });
    try {
      const deployments = await listSapDeploymentsCall(accessToken, serviceKey, resourceGroup);
      setState({ status: "loaded", deployments });
    } catch (error) {
      setState({
        status: "error",
        message: error instanceof Error ? error.message : "Failed to discover deployments.",
      });
    }
  };

  const selectDeployment = (deployment: SapDeploymentInfo) => {
    const selection = buildSapDeploymentSelection(deployment);
    form.setValue("model", selection.model);
    form.setValue("model_mappings", selection.model_mappings);
    form.setValue("api_base", selection.api_base);
    setSelectedId(deployment.id);
  };

  const selectedModelName =
    state.status === "loaded" ? state.deployments.find((d) => d.id === selectedId)?.model_name : undefined;

  return (
    <div className="mb-4">
      <MountedFormField
        name="resource_group"
        label="AI Resource Group"
        help="SAP AI Core resource group for discovery and inference. Leave blank to use 'default'."
        className="mb-4"
      >
        {(control) => (
          <Input
            id={control.id}
            value={(control.value as string | undefined) ?? undefined}
            onBlur={control.onBlur}
            onChange={control.onChange}
            placeholder="default"
            type="text"
          />
        )}
      </MountedFormField>
      <Button
        type="button"
        variant="outline"
        className="w-fit"
        onClick={discover}
        disabled={state.status === "loading"}
      >
        {state.status === "loading" ? "Discovering..." : "Discover deployments"}
      </Button>
      {state.status === "error" && <p className="text-sm mt-2 text-destructive">{state.message}</p>}
      {state.status === "loaded" && state.deployments.length === 0 && (
        <p className="text-sm mt-2">No running deployments found for this service key.</p>
      )}
      {state.status === "loaded" && state.deployments.length > 0 && (
        <>
          <ul className="mt-2 border rounded-md divide-y">
            {state.deployments.map((deployment) => {
              const isSelected = deployment.id === selectedId;
              return (
                <li key={deployment.id}>
                  <button
                    type="button"
                    aria-pressed={isSelected}
                    className={`flex w-full items-start gap-2 px-3 py-2 text-left text-sm ${
                      isSelected ? "bg-primary/10 text-primary" : "hover:bg-muted"
                    }`}
                    onClick={() => selectDeployment(deployment)}
                  >
                    <Check
                      className={`mt-0.5 h-4 w-4 shrink-0 ${isSelected ? "opacity-100" : "opacity-0"}`}
                      aria-hidden
                    />
                    <span className="min-w-0">
                      <span className="font-mono">{deployment.model_name}</span>
                      <span className="block text-xs text-muted-foreground">{deployment.deployment_url}</span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
          {selectedModelName !== undefined && (
            <p className="mt-2 text-sm text-muted-foreground">
              Selected <span className="font-mono">{selectedModelName}</span>. Model fields above are filled in.
            </p>
          )}
        </>
      )}
    </div>
  );
};

export default SapDeploymentDiscovery;
