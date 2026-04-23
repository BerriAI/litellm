import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { LogEntry } from "../columns";
import { parseToolsFromLog } from "./utils";
import { ToolItem } from "./ToolItem";

interface ToolsSectionProps {
  log: LogEntry;
}

export function ToolsSection({ log }: ToolsSectionProps) {
  const tools = parseToolsFromLog(log);

  if (tools.length === 0) return null;

  const totalTools = tools.length;
  const calledTools = tools.filter((t) => t.called).length;

  const toolNamePreview = tools
    .slice(0, 2)
    .map((t) => t.name)
    .join(", ");
  const hasMoreTools = tools.length > 2;

  return (
    <div className="bg-background rounded-lg shadow w-full max-w-full overflow-hidden mb-6">
      <Accordion type="single" collapsible>
        <AccordionItem value="tools">
          <AccordionTrigger>
            <div className="flex items-center gap-3 flex-wrap pr-2">
              <h3 className="text-lg font-medium text-foreground m-0">
                Tools
              </h3>
              <span className="text-sm text-muted-foreground">
                {totalTools} provided, {calledTools} called
              </span>
              <span className="text-sm text-muted-foreground">
                • {toolNamePreview}
                {hasMoreTools && "..."}
              </span>
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="flex flex-col gap-2">
              {tools.map((tool) => (
                <ToolItem key={tool.name} tool={tool} />
              ))}
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}
