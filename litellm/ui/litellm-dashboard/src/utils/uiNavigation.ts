import { getUiPageUrl } from "@/components/networking";

export function navigateToUiPage(url: string, replace = false): void {
  let destination = "/ui";

  try {
    const target = new URL(getUiPageUrl(url), window.location.origin);
    const isHttp = target.protocol === "http:" || target.protocol === "https:";

    if (isHttp && target.origin === window.location.origin) {
      destination = target.href;
    }
  } catch {
    destination = "/ui";
  }

  if (replace) {
    window.location.replace(destination);
  } else {
    // eslint-disable-next-line @next/next/no-location-assign-relative-destination -- A fresh document resets auth state after login.
    window.location.assign(destination);
  }
}
