import { Loader2 } from "lucide-react";

import { StateMessage, type StateMessageProps } from "./StateMessage";

type LoadingStateProps = Omit<StateMessageProps, "role" | "icon" | "tone">;

export function LoadingState({ title, ...props }: LoadingStateProps) {
  return (
    <StateMessage
      role="status"
      aria-label={title}
      icon={<Loader2 className="size-5 animate-spin motion-reduce:animate-none" />}
      title={title}
      {...props}
    />
  );
}
