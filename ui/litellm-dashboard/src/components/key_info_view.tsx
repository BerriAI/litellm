import React from "react";
import { Card, Text, Button, Grid, Col } from "@tremor/react";
import { KeyResponse } from "./all_keys_table";
import { ArrowLeftIcon } from "@heroicons/react/outline";

interface KeyInfoViewProps {
  keyId: string;
  onClose: () => void;
  keyData: KeyResponse | undefined;
}

export default function KeyInfoView({ keyId, onClose, keyData }: KeyInfoViewProps) {
  if (!keyData) {
    return (
      <div className="p-4">
        <Button 
          icon={ArrowLeftIcon} 
          variant="light"
          onClick={onClose}
          className="mb-4"
        >
          Back to Keys
        </Button>
        <Text>Key not found</Text>
      </div>
    );
  }

  return (
    <div className="p-4">
      <Button 
        icon={ArrowLeftIcon} 
        variant="light"
        onClick={onClose}
        className="mb-4"
      >
        Back to Keys
      </Button>

      <Grid numItems={1} className="gap-4">
        <Col>
          <Card>
            <h3 className="text-lg font-medium mb-4">Key Details</h3>
            <div className="space-y-4">
              <div>
                <Text className="font-medium">Key ID</Text>
                <Text className="font-mono">{keyData.token}</Text>
              </div>
              
              <div>
                <Text className="font-medium">Key Alias</Text>
                <Text>{keyData.key_alias || "Not Set"}</Text>
              </div>

              <div>
                <Text className="font-medium">Secret Key</Text>
                <Text className="font-mono">{keyData.key_name}</Text>
              </div>

              <div>
                <Text className="font-medium">Team ID</Text>
                <Text>{keyData.team_id || "Not Set"}</Text>
              </div>

              <div>
                <Text className="font-medium">Organization</Text>
                <Text>{keyData.organization_id || "Not Set"}</Text>
              </div>

              <div>
                <Text className="font-medium">Created</Text>
                <Text>{new Date(keyData.created_at).toLocaleString()}</Text>
              </div>

              <div>
                <Text className="font-medium">Expires</Text>
                <Text>{keyData.expires ? new Date(keyData.expires).toLocaleString() : "Never"}</Text>
              </div>

              <div>
                <Text className="font-medium">Spend</Text>
                <Text>${Number(keyData.spend).toFixed(4)} USD</Text>
              </div>

              <div>
                <Text className="font-medium">Budget</Text>
                <Text>{keyData.max_budget !== null ? `$${keyData.max_budget} USD` : "Unlimited"}</Text>
              </div>

              <div>
                <Text className="font-medium">Models</Text>
                <div className="flex flex-wrap gap-2 mt-1">
                  {keyData.models && keyData.models.length > 0 ? (
                    keyData.models.map((model, index) => (
                      <span
                        key={index}
                        className="px-2 py-1 bg-blue-100 rounded text-xs"
                      >
                        {model}
                      </span>
                    ))
                  ) : (
                    <Text>No models specified</Text>
                  )}
                </div>
              </div>

              <div>
                <Text className="font-medium">Rate Limits</Text>
                <Text>TPM: {keyData.tpm_limit !== null ? keyData.tpm_limit : "Unlimited"}</Text>
                <Text>RPM: {keyData.rpm_limit !== null ? keyData.rpm_limit : "Unlimited"}</Text>
              </div>

              <div>
                <Text className="font-medium">Metadata</Text>
                <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto mt-1">
                  {JSON.stringify(keyData.metadata, null, 2)}
                </pre>
              </div>
            </div>
          </Card>
        </Col>
      </Grid>
    </div>
  );
} 