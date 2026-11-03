import { useState } from "react";
import { Check, ChevronDown, ImageIcon, MessageSquare, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";

export type ChatMode = "chat" | "image";

export function modelChatMode(model: ModelGroup): ChatMode {
  return model.mode === "image_generation" ? "image" : "chat";
}

export function isChatModel(model: ModelGroup): boolean {
  return !model.mode || ["chat", "responses", "completion", "image_generation"].includes(model.mode);
}

export default function ChatModelPicker({
  models,
  selectedModel,
  loading,
  disabled,
  onSelect,
}: {
  models: ModelGroup[];
  selectedModel: string | null;
  loading: boolean;
  disabled: boolean;
  onSelect: (model: ModelGroup) => void;
}) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | ChatMode>("all");
  const selectModel = (model: ModelGroup) => {
    onSelect(model);
    setOpen(false);
    setSearch("");
  };
  const visibleModels = models.filter(
    (model) =>
      (filter === "all" || modelChatMode(model) === filter) &&
      `${model.model_group} ${(model.providers ?? []).join(" ")}`.toLowerCase().includes(search.toLowerCase()),
  );

  if (loading) return <Skeleton className="h-9 w-48" />;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        render={
          <Button variant="ghost" disabled={disabled} className="moyai-chat-model-picker">
            <span className="truncate">{selectedModel ?? "Choose a model"}</span>
            <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" />
          </Button>
        }
      />
      <PopoverContent align="start" className="chat-theme w-[min(360px,calc(100vw-2rem))] p-0">
        <div className="space-y-3 p-3">
          <div>
            <p className="text-sm font-medium">Model Hub</p>
            <p className="text-xs text-muted-foreground">Choose a model to try it here.</p>
          </div>
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 size-4 text-muted-foreground" />
            <Input
              aria-label="Search models"
              placeholder="Search models or providers"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              className="pl-8"
            />
          </div>
          <div className="flex gap-1" aria-label="Model type">
            {(
              [
                ["all", "All"],
                ["chat", "Chat"],
                ["image", "Images"],
              ] as const
            ).map(([value, label]) => (
              <Button
                key={value}
                size="sm"
                variant={filter === value ? "secondary" : "ghost"}
                aria-pressed={filter === value}
                onClick={() => setFilter(value)}
              >
                {label}
              </Button>
            ))}
          </div>
        </div>
        <ScrollArea className="h-[min(320px,50dvh)] border-t p-1">
          {visibleModels.length === 0 && <p className="p-4 text-sm text-muted-foreground">No models found.</p>}
          {visibleModels.map((model) => {
            const image = modelChatMode(model) === "image";
            const Icon = image ? ImageIcon : MessageSquare;
            return (
              <Button
                key={model.model_group}
                variant="ghost"
                className="h-auto w-full justify-start gap-3 py-3"
                onClick={() => selectModel(model)}
              >
                <Icon className="size-4 shrink-0 text-muted-foreground" />
                <span className="min-w-0 flex-1 text-left">
                  <span className="block truncate text-sm">{model.model_group}</span>
                  <span className="block truncate text-xs font-normal text-muted-foreground">
                    {[image ? "Image generation" : "Chat", ...(model.providers ?? [])].join(" · ")}
                  </span>
                </span>
                {selectedModel === model.model_group && <Check className="size-4 shrink-0 text-primary" />}
              </Button>
            );
          })}
        </ScrollArea>
      </PopoverContent>
    </Popover>
  );
}
