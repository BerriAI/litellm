import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ArrowLeft,
  Calendar,
  Check,
  Clock,
  Copy,
  Mail,
  Plus,
  RefreshCcw,
  Shield,
  Trash2,
  User,
  Wallet,
  Zap,
} from "lucide-react";
import LabeledField from "../common_components/LabeledField";

export interface KeyInfoData {
  keyName: string;
  keyId: string;
  userId: string;
  userEmail: string;
  createdBy: string;
  createdAt: string;
  lastUpdated: string;
  lastActive: string;
}

interface KeyInfoHeaderProps {
  data: KeyInfoData;
  onBack?: () => void;
  onCreateNew?: () => void;
  onRegenerate?: () => void;
  onDelete?: () => void;
  onResetSpend?: () => void;
  canModifyKey?: boolean;
  backButtonText?: string;
  regenerateDisabled?: boolean;
  regenerateTooltip?: string;
}

const CopyableInline: React.FC<{
  text: string;
  label: string;
  copyTooltip: string;
}> = ({ text, label, copyTooltip }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (_err) {
      /* noop */
    }
  };
  return (
    <span className="inline-flex items-center gap-1">
      <span>{label}</span>
      <TooltipProvider>
        <Tooltip open={copied || undefined}>
          <TooltipTrigger asChild>
            <button
              type="button"
              onClick={handleCopy}
              className="text-muted-foreground hover:text-foreground"
              aria-label={copyTooltip}
            >
              {copied ? (
                <Check className="h-3 w-3 text-emerald-500" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent>{copied ? "Copied!" : copyTooltip}</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </span>
  );
};

export function KeyInfoHeader({
  data,
  onBack,
  onCreateNew,
  onRegenerate,
  onDelete,
  onResetSpend,
  canModifyKey = true,
  backButtonText = "Back to Keys",
  regenerateDisabled = false,
  regenerateTooltip,
}: KeyInfoHeaderProps) {
  return (
    <div>
      {onCreateNew && (
        <div className="mb-4">
          <Button onClick={onCreateNew}>
            <Plus className="h-4 w-4" />
            Create New Key
          </Button>
        </div>
      )}

      <div className="mb-4">
        <Button variant="ghost" onClick={onBack}>
          <ArrowLeft className="h-4 w-4" />
          {backButtonText}
        </Button>
      </div>

      <div className="flex justify-between items-start mb-5">
        <div>
          <h3 className="text-xl font-semibold m-0">
            <CopyableInline
              text={data.keyName}
              label={data.keyName}
              copyTooltip="Copy Key Alias"
            />
          </h3>
          <div className="text-sm text-muted-foreground">
            <CopyableInline
              text={data.keyId}
              label={`Key ID: ${data.keyId}`}
              copyTooltip="Copy Key ID"
            />
          </div>
        </div>
        {canModifyKey && (
          <div className="flex gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span>
                    <Button
                      variant="outline"
                      onClick={onRegenerate}
                      disabled={regenerateDisabled}
                    >
                      <RefreshCcw className="h-4 w-4" />
                      Regenerate Key
                    </Button>
                  </span>
                </TooltipTrigger>
                {regenerateTooltip && (
                  <TooltipContent>{regenerateTooltip}</TooltipContent>
                )}
              </Tooltip>
            </TooltipProvider>
            {onResetSpend && (
              <Button
                variant="outline"
                className="text-destructive border-destructive/30 hover:bg-destructive/10"
                onClick={onResetSpend}
              >
                <Wallet className="h-4 w-4" />
                Reset Spend
              </Button>
            )}
            <Button
              variant="outline"
              className="text-destructive border-destructive/30 hover:bg-destructive/10"
              onClick={onDelete}
            >
              <Trash2 className="h-4 w-4" />
              Delete Key
            </Button>
          </div>
        )}
      </div>

      <div className="flex items-stretch gap-10 mb-10">
        <div className="flex flex-col gap-4">
          <LabeledField
            label="User Email"
            value={data.userEmail}
            icon={<Mail className="h-3 w-3" />}
          />
          <LabeledField
            label="User ID"
            value={data.userId}
            icon={<User className="h-3 w-3" />}
            truncate
            copyable
            defaultUserIdCheck
          />
        </div>

        <div className="w-px bg-border" />

        <div className="flex flex-col gap-4">
          <LabeledField
            label="Created At"
            value={data.createdAt}
            icon={<Calendar className="h-3 w-3" />}
          />
          <LabeledField
            label="Created By"
            value={data.createdBy}
            icon={<Shield className="h-3 w-3" />}
            truncate
            copyable
            defaultUserIdCheck
          />
        </div>

        <div className="w-px bg-border" />

        <div className="flex flex-col gap-4">
          <LabeledField
            label="Last Updated"
            value={data.lastUpdated}
            icon={<Clock className="h-3 w-3" />}
          />
          <LabeledField
            label="Last Active"
            value={data.lastActive}
            icon={<Zap className="h-3 w-3" />}
          />
        </div>
      </div>
    </div>
  );
}
