import { getUiPageUrl } from "@/components/networking";

export function navigateToUiPage(url: string, replace = false): void {
  const destination = (() => {
    try {
      const target = new URL(getUiPageUrl(url), window.location.origin);
      const isHttp = target.protocol === "http:" || target.protocol === "https:";

      return isHttp && target.origin === window.location.origin ? target.href : "/ui";
    } catch {
      return "/ui";
    }
  })();

  if (replace) {
    window.location.replace(destination);
  } else {
    window.location.assign(destination);
  }
}
