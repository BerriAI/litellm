import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Info, AlertCircle } from "lucide-react"

interface LoginFormProps extends React.ComponentPropsWithoutRef<"form"> {
  username?: string;
  password?: string;
  setUsername?: (value: string) => void;
  setPassword?: (value: string) => void;
  onSubmit?: (e: React.FormEvent) => void;
  isLoading?: boolean;
  error?: string | null;
}

export function LoginForm({
  className,
  username,
  password,
  setUsername,
  setPassword,
  onSubmit,
  isLoading,
  error,
  ...props
}: LoginFormProps) {
  return (
    <form className={cn("flex flex-col gap-6", className)} onSubmit={onSubmit} {...props}>
      <div className="flex flex-col items-center gap-2 text-center">
        <h1 className="text-2xl font-bold">Login to LiteLLM</h1>
        <p className="text-balance text-sm text-muted-foreground">
          Enter your credentials ensuring access to the simplified Admin UI
        </p>
      </div>
      <div className="grid gap-6">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <Alert>
          <Info className="h-4 w-4" />
          <AlertTitle>Default Credentials</AlertTitle>
          <AlertDescription>
            Username: <code className="bg-muted px-1 py-0.5 rounded text-xs">admin</code><br />
            Password: Your Proxy <code className="bg-muted px-1 py-0.5 rounded text-xs">MASTER_KEY</code>
          </AlertDescription>
        </Alert>

        <div className="grid gap-2">
          <Label htmlFor="username">Username</Label>
          <Input
            id="username"
            type="text"
            placeholder="admin"
            required
            value={username}
            onChange={(e) => setUsername?.(e.target.value)}
            disabled={isLoading}
          />
        </div>
        <div className="grid gap-2">
          <div className="flex items-center">
            <Label htmlFor="password">Password</Label>
          </div>
          <Input
            id="password"
            type="password"
            required
            value={password}
            onChange={(e) => setPassword?.(e.target.value)}
            disabled={isLoading}
          />
        </div>
        <Button type="submit" className="w-full" disabled={isLoading}>
          {isLoading ? "Logging in..." : "Login"}
        </Button>
      </div>
      <div className="text-center text-sm">
        Need help?{" "}
        <a href="https://docs.litellm.ai/docs/proxy/ui" target="_blank" rel="noopener noreferrer" className="underline underline-offset-4">
          Documentation
        </a>
      </div>
    </form>
  )
}
