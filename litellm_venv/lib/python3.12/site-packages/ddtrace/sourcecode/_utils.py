import re
import subprocess

from ddtrace.internal.compat import parse


SCP_REGEXP = re.compile("^[a-z0-9_]+@([a-z0-9._-]+):(.*)$", re.IGNORECASE)


def _remove_suffix(s, suffix):
    if s.endswith(suffix):
        return s[: -len(suffix)]
    else:
        return s


def normalize_repository_url(url):
    scheme = ""
    hostname = ""
    port = None
    path = ""

    match = SCP_REGEXP.match(url)
    if match:
        # Check URLs like "git@github.com:user/project.git",
        scheme = "https"
        hostname = match.group(1)
        path = "/" + match.group(2)
    else:
        u = parse.urlsplit(url)
        if u.scheme == "" and u.hostname is None:
            # Try to add a scheme.
            u = parse.urlsplit("https://" + url)  # Default to HTTPS.
            if u.hostname is None:
                return ""

        scheme = u.scheme
        hostname = u.hostname
        port = u.port
        path = u.path

        if scheme not in ("http", "https", "git", "ssh"):
            return ""

        if not scheme.startswith("http"):
            scheme = "https"  # Default to HTTPS.
            port = None

    path = _remove_suffix(path, ".git/")
    path = _remove_suffix(path, ".git")

    netloc = hostname
    if port is not None:
        netloc += ":" + str(port)

    return parse.urlunsplit((scheme, netloc, path, "", ""))


def _query_git(args):
    ver = subprocess.Popen(["git"] + args, stdout=subprocess.PIPE).communicate()[0]
    return ver.strip().decode("utf-8")


def get_commit_sha():
    return _query_git(["rev-parse", "HEAD"])


def get_repository_url():
    return _query_git(["config", "--get", "remote.origin.url"])


def get_source_code_link():
    return normalize_repository_url(get_repository_url()) + "#" + get_commit_sha()
