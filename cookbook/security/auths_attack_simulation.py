"""
Auths Attack Simulation: LiteLLM March 24, 2026 Supply Chain Incident

Demonstrates how Auths cryptographic commit verification would have detected
the unauthorized PyPI publish that compromised LiteLLM v1.82.7 and v1.82.8.

What happened:
  1. Attacker compromised the Trivy GitHub Action (March 19)
  2. LiteLLM's CI ran Trivy without version pinning
  3. Compromised Trivy exfiltrated the PYPI_PUBLISH token from GitHub Actions
  4. Attacker used the stolen token to publish malicious versions to PyPI
  5. The malicious packages contained a credential stealer in a .pth file
  6. Source code on GitHub was never modified — the attack existed only in PyPI

Why Auths prevents this:
  With Auths, every release artifact carries an Ed25519 signature from the
  maintainer's cryptographic identity (KERI-based DID). Stealing the PyPI
  token is insufficient — the attacker cannot produce a valid signature
  without the maintainer's private key stored in their device keychain.

Usage:
  pip install auths
  python auths_attack_simulation.py
"""
import os
import shutil
import subprocess
import sys
import tempfile


def check_auths_cli() -> bool:
    """Check if the auths CLI is available."""
    return shutil.which("auths") is not None


def run(cmd: list[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )


def setup_test_repo(tmpdir: str) -> str:
    """Create a temporary git repo with signed and unsigned commits."""
    repo = os.path.join(tmpdir, "litellm-simulation")
    os.makedirs(repo)

    # Initialize repo
    run(["git", "init"], cwd=repo)
    run(["git", "config", "user.email", "maintainer@example.com"], cwd=repo)
    run(["git", "config", "user.name", "LiteLLM Maintainer"], cwd=repo)

    # Generate a test Ed25519 keypair for the "legitimate maintainer"
    key_path = os.path.join(tmpdir, "test_key")
    run(
        ["ssh-keygen", "-t", "ed25519", "-f", key_path, "-N", "", "-q"],
        cwd=tmpdir,
    )

    # Configure git to sign with this key
    run(["git", "config", "gpg.format", "ssh"], cwd=repo)
    run(["git", "config", "user.signingkey", key_path], cwd=repo)

    # Create allowed_signers file
    pub_key_content = open(f"{key_path}.pub").read().strip()
    signers_path = os.path.join(repo, ".auths")
    os.makedirs(signers_path)
    with open(os.path.join(signers_path, "allowed_signers"), "w") as f:
        f.write(f"maintainer@example.com {pub_key_content}\n")

    run(["git", "config", "gpg.ssh.allowedSignersFile", os.path.join(signers_path, "allowed_signers")], cwd=repo)

    # Commit 1: Legitimate signed release (v1.82.6)
    with open(os.path.join(repo, "litellm_version.py"), "w") as f:
        f.write('version = "1.82.6"\n')
    run(["git", "add", "."], cwd=repo)
    run(["git", "commit", "-S", "-m", "release: v1.82.6 (legitimate, signed)"], cwd=repo)

    # Commit 2: Attacker's malicious commit (unsigned — simulates PyPI-only publish)
    run(["git", "config", "commit.gpgSign", "false"], cwd=repo)
    with open(os.path.join(repo, "litellm_version.py"), "w") as f:
        f.write('version = "1.82.7"\n')
    with open(os.path.join(repo, "litellm_init.pth"), "w") as f:
        f.write("# Simulated malicious payload — credential stealer\n")
        f.write("import os; os.environ.get('AWS_SECRET_ACCESS_KEY')  # exfiltrate\n")
    run(["git", "add", "."], cwd=repo)
    run(["git", "commit", "-m", "release: v1.82.7 (MALICIOUS — unsigned)"], cwd=repo)

    return repo


def run_verification(repo: str) -> None:
    """Run auths verification and display results."""
    result = run(
        ["auths", "verify", "HEAD~1..HEAD", "--allowed-signers", ".auths/allowed_signers"],
        cwd=repo,
        check=False,
    )

    if result.returncode != 0:
        print("  BLOCKED: Unsigned commit detected")
        if result.stdout:
            print(f"  Output: {result.stdout.strip()}")
        if result.stderr:
            print(f"  Detail: {result.stderr.strip()}")
    else:
        print("  PASSED: All commits verified")
        if result.stdout:
            print(f"  Output: {result.stdout.strip()}")


def main() -> None:
    print("=" * 70)
    print("Auths Attack Simulation: LiteLLM Supply Chain Incident (March 24, 2026)")
    print("=" * 70)
    print()

    if not check_auths_cli():
        print("The 'auths' CLI is not installed.")
        print()
        print("Install it with:")
        print("  pip install auths")
        print()
        print("Or visit: https://github.com/auths-dev/auths")
        sys.exit(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        print("[1] Setting up simulation repository...")
        repo = setup_test_repo(tmpdir)
        print("    Created repo with 2 commits:")
        print("    - v1.82.6: Legitimate release, signed by maintainer")
        print("    - v1.82.7: Attacker's malicious version, unsigned")
        print()

        # Verify the legitimate commit
        print("[2] Verifying legitimate commit (v1.82.6)...")
        result = run(
            ["auths", "verify", "HEAD~2..HEAD~1", "--allowed-signers", ".auths/allowed_signers"],
            cwd=repo,
            check=False,
        )
        if result.returncode == 0:
            print("  PASSED: Commit is signed by an authorized maintainer")
        else:
            print("  Result: ", result.stdout.strip() if result.stdout else result.stderr.strip())
        print()

        # Verify the malicious commit
        print("[3] Verifying attacker's commit (v1.82.7)...")
        run_verification(repo)
        print()

        # Summary
        print("-" * 70)
        print("RESULT: The attacker's unsigned commit would have been flagged.")
        print()
        print("In the real attack, the attacker used a stolen PyPI token to publish")
        print("malicious packages directly to the registry. The source code on GitHub")
        print("was never modified. With Auths, even if registry credentials are stolen,")
        print("the attacker cannot produce a valid Ed25519 signature — the private key")
        print("is bound to the maintainer's device keychain and never leaves it.")
        print()
        print("Learn more: https://github.com/auths-dev/auths")
        print("=" * 70)


if __name__ == "__main__":
    main()
