"use client";

import { ChevronDown, FileText, Music, X } from "lucide-react";
import { EndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import {
  Attachment,
  AttachmentAction,
  AttachmentActions,
  AttachmentContent,
  AttachmentDescription,
  AttachmentGroup,
  AttachmentMedia,
  AttachmentTitle,
} from "@/components/ui/attachment";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { InputGroupButton } from "@/components/ui/input-group";
import { ENDPOINT_OPTIONS, OPEN_AI_VOICE_SELECT_OPTIONS, type OpenAIVoice } from "../chat_ui/chatConstants";
import { formatBytes, type DraftFile } from "./composerFiles";

const shortEndpoint = (label: string): string => label.replace(/^\/v1(beta)?\//, "");

export function EndpointMenu({
  value,
  onValueChange,
}: {
  value: EndpointType;
  onValueChange: (value: EndpointType) => void;
}) {
  const current = ENDPOINT_OPTIONS.find((option) => option.value === value);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={<InputGroupButton size="xs" className="shrink-0" aria-label="Endpoint" />}>
        <span className="font-mono text-xs">{shortEndpoint(current?.label ?? value)}</span>
        <ChevronDown className="text-muted-foreground" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-64">
        <DropdownMenuRadioGroup value={value} onValueChange={(next) => onValueChange(next as EndpointType)}>
          <DropdownMenuLabel>Endpoint</DropdownMenuLabel>
          {ENDPOINT_OPTIONS.map((option) => (
            <DropdownMenuRadioItem key={option.value} value={option.value} closeOnClick className="font-mono text-xs">
              {option.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function VoiceMenu({
  value,
  onValueChange,
}: {
  value: OpenAIVoice;
  onValueChange: (value: OpenAIVoice) => void;
}) {
  const current = OPEN_AI_VOICE_SELECT_OPTIONS.find((option) => option.value === value);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={<InputGroupButton size="xs" className="shrink-0" aria-label="Voice" />}>
        <span className="text-xs">Voice: {current?.label.split(" - ")[0] ?? value}</span>
        <ChevronDown className="text-muted-foreground" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-72">
        <DropdownMenuRadioGroup value={value} onValueChange={(next) => onValueChange(next as OpenAIVoice)}>
          <DropdownMenuLabel>Voice</DropdownMenuLabel>
          {OPEN_AI_VOICE_SELECT_OPTIONS.map((option) => (
            <DropdownMenuRadioItem key={option.value} value={option.value} closeOnClick>
              {option.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function FileMedia({ draft }: { draft: DraftFile }) {
  if (draft.previewUrl)
    return (
      <AttachmentMedia variant="image">
        <img src={draft.previewUrl} alt="" />
      </AttachmentMedia>
    );
  return <AttachmentMedia>{draft.file.type.startsWith("audio/") ? <Music /> : <FileText />}</AttachmentMedia>;
}

export function DraftAttachments({
  files,
  onRemove,
}: {
  files: readonly DraftFile[];
  onRemove: (index: number) => void;
}) {
  return (
    <AttachmentGroup className="w-full">
      {files.map((draft, index) => (
        <Attachment key={`${draft.file.name}-${index}`} size="xs">
          <FileMedia draft={draft} />
          <AttachmentContent>
            <AttachmentTitle>{draft.file.name}</AttachmentTitle>
            <AttachmentDescription>{formatBytes(draft.file.size)}</AttachmentDescription>
          </AttachmentContent>
          <AttachmentActions>
            <AttachmentAction aria-label={`Remove ${draft.file.name}`} onClick={() => onRemove(index)}>
              <X />
            </AttachmentAction>
          </AttachmentActions>
        </Attachment>
      ))}
    </AttachmentGroup>
  );
}
