import React, { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { Check } from "lucide-react";
import {
  enableClaudeCodePlugin,
  disableClaudeCodePlugin,
} from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import { Plugin } from "./types";

interface MakeSkillPublicFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string;
  skillsList: Plugin[];
  onSuccess: () => void;
}

function Stepper({ current, steps }: { current: number; steps: string[] }) {
  return (
    <ol className="flex items-center gap-2 mb-6">
      {steps.map((label, i) => {
        const active = i === current;
        const completed = i < current;
        return (
          <li
            key={label}
            className="flex items-center gap-2 flex-1 min-w-0"
          >
            <div
              className={cn(
                "flex h-6 w-6 items-center justify-center rounded-full border text-xs font-medium",
                completed
                  ? "bg-primary text-primary-foreground border-primary"
                  : active
                    ? "border-primary text-primary"
                    : "border-border text-muted-foreground",
              )}
            >
              {completed ? <Check className="h-3 w-3" /> : i + 1}
            </div>
            <span
              className={cn(
                "text-sm truncate",
                active || completed
                  ? "text-foreground"
                  : "text-muted-foreground",
              )}
            >
              {label}
            </span>
            {i < steps.length - 1 && (
              <div className="h-px flex-1 bg-border" />
            )}
          </li>
        );
      })}
    </ol>
  );
}

const MakeSkillPublicForm: React.FC<MakeSkillPublicFormProps> = ({
  visible,
  onClose,
  accessToken,
  skillsList,
  onSuccess,
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedSkills, setSelectedSkills] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);

  const handleClose = () => {
    setCurrentStep(0);
    setSelectedSkills(new Set());
    onClose();
  };

  const handleNext = () => {
    if (selectedSkills.size === 0) {
      NotificationsManager.fromBackend("Please select at least one skill");
      return;
    }
    setCurrentStep(1);
  };

  const handleSkillSelection = (name: string, checked: boolean) => {
    const next = new Set(selectedSkills);
    if (checked) {
      next.add(name);
    } else {
      next.delete(name);
    }
    setSelectedSkills(next);
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedSkills(new Set(skillsList.map((s) => s.name)));
    } else {
      setSelectedSkills(new Set());
    }
  };

  useEffect(() => {
    if (visible && skillsList.length > 0) {
      setSelectedSkills(
        new Set(skillsList.filter((s) => s.enabled).map((s) => s.name)),
      );
    }
  }, [visible, skillsList]);

  const handleSubmit = async () => {
    if (selectedSkills.size === 0) {
      NotificationsManager.fromBackend("Please select at least one skill");
      return;
    }

    setLoading(true);
    try {
      const selectedSet = selectedSkills;
      await Promise.all(
        skillsList.map((skill) => {
          const shouldBePublic = selectedSet.has(skill.name);
          if (shouldBePublic && !skill.enabled) {
            return enableClaudeCodePlugin(accessToken, skill.name);
          }
          if (!shouldBePublic && skill.enabled) {
            return disableClaudeCodePlugin(accessToken, skill.name);
          }
          return Promise.resolve();
        }),
      );

      NotificationsManager.success(
        `Skill Hub updated — ${selectedSkills.size} skill(s) published`,
      );
      handleClose();
      onSuccess();
    } catch (error) {
      console.error("Error publishing skills:", error);
      NotificationsManager.fromBackend(
        "Failed to update skills. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  };

  const allSelected =
    skillsList.length > 0 && skillsList.every((s) => selectedSkills.has(s.name));
  const isIndeterminate = selectedSkills.size > 0 && !allSelected;

  const renderStep1 = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Select Skills to Publish</h3>
        <label className="flex items-center gap-2 cursor-pointer">
          <Checkbox
            checked={
              isIndeterminate ? "indeterminate" : allSelected ? true : false
            }
            onCheckedChange={(c) => handleSelectAll(c === true)}
            disabled={skillsList.length === 0}
          />
          <span className="text-sm">Select All ({skillsList.length})</span>
        </label>
      </div>

      <p className="text-sm text-muted-foreground">
        Selected skills will be visible to all users in the Skill Hub.
        Deselected skills will be unpublished.
      </p>

      <div className="max-h-96 overflow-y-auto border border-border rounded-lg p-4">
        <div className="space-y-3">
          {skillsList.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <p>No skills registered yet.</p>
            </div>
          ) : (
            skillsList.map((skill) => (
              <label
                key={skill.name}
                className="flex items-center space-x-3 p-3 border border-border rounded-lg hover:bg-muted cursor-pointer"
              >
                <Checkbox
                  checked={selectedSkills.has(skill.name)}
                  onCheckedChange={(c) =>
                    handleSkillSelection(skill.name, c === true)
                  }
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium font-mono text-sm">
                      {skill.name}
                    </span>
                    {skill.enabled && (
                      <Badge className="text-xs bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
                        Public
                      </Badge>
                    )}
                  </div>
                  {skill.description && (
                    <span className="text-xs text-muted-foreground truncate max-w-sm block">
                      {skill.description}
                    </span>
                  )}
                </div>
                {skill.domain && (
                  <Badge className="text-xs bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                    {skill.domain}
                  </Badge>
                )}
              </label>
            ))
          )}
        </div>
      </div>

      {selectedSkills.size > 0 && (
        <div className="bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 rounded-lg p-3">
          <p className="text-sm text-blue-800 dark:text-blue-200">
            <strong>{selectedSkills.size}</strong> skill
            {selectedSkills.size !== 1 ? "s" : ""} will be published
          </p>
        </div>
      )}
    </div>
  );

  const renderStep2 = () => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Confirm Publish to Skill Hub</h3>

      <div className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900 rounded-lg p-4">
        <p className="text-sm text-amber-800 dark:text-amber-200">
          <strong>Note:</strong> Published skills will be visible to all users
          in the Skill Hub tab. Skills not in the list below will be unpublished.
        </p>
      </div>

      <div className="space-y-3">
        <p className="font-medium">Skills to be published:</p>
        <div className="max-h-48 overflow-y-auto border border-border rounded-lg p-3">
          <div className="space-y-2">
            {Array.from(selectedSkills).map((name) => {
              const skill = skillsList.find((s) => s.name === name);
              return (
                <div
                  key={name}
                  className="flex items-center justify-between p-2 bg-muted rounded"
                >
                  <span className="font-mono text-sm">{name}</span>
                  {skill?.domain && (
                    <Badge className="text-xs bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                      {skill.domain}
                    </Badge>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 rounded-lg p-3">
        <p className="text-sm text-blue-800 dark:text-blue-200">
          Total: <strong>{selectedSkills.size}</strong> skill
          {selectedSkills.size !== 1 ? "s" : ""} will be published
        </p>
      </div>
    </div>
  );

  return (
    <Dialog
      open={visible}
      onOpenChange={(o) => (!o ? handleClose() : undefined)}
    >
      <DialogContent className="max-w-[700px]">
        <DialogHeader>
          <DialogTitle>Publish to Skill Hub</DialogTitle>
        </DialogHeader>
        <Stepper current={currentStep} steps={["Select Skills", "Confirm"]} />

        {currentStep === 0 ? renderStep1() : renderStep2()}

        <div className="flex justify-between mt-6">
          <Button
            variant="outline"
            onClick={
              currentStep === 0 ? handleClose : () => setCurrentStep(0)
            }
          >
            {currentStep === 0 ? "Cancel" : "Previous"}
          </Button>
          <div className="flex space-x-2">
            {currentStep === 0 && (
              <Button
                onClick={handleNext}
                disabled={selectedSkills.size === 0}
              >
                Next
              </Button>
            )}
            {currentStep === 1 && (
              <Button onClick={handleSubmit} disabled={loading}>
                {loading ? "Publishing..." : "Publish to Hub"}
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default MakeSkillPublicForm;
