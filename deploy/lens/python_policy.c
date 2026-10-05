#include <errno.h>
#include <fcntl.h>
#include <seccomp.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>

static int allow(scmp_filter_ctx policy, const char *name)
{
    int number = seccomp_syscall_resolve_name(name);
    return number < 0 ? 0 : seccomp_rule_add(policy, SCMP_ACT_ALLOW, number, 0);
}

int main(int argc, char **argv)
{
    const char *calls[] = {
        "read", "write", "readv", "writev", "pread64", "pwrite64", "close", "close_range",
        "open", "openat", "openat2", "fstat", "stat", "lstat", "newfstatat", "statx",
        "lseek", "getdents", "getdents64", "access", "faccessat", "faccessat2",
        "readlink", "readlinkat", "getcwd", "chdir", "fchdir", "statfs", "fstatfs",
        "mkdir", "mkdirat", "rmdir", "unlink", "unlinkat", "rename", "renameat", "renameat2",
        "link", "linkat", "symlink", "symlinkat", "truncate", "ftruncate", "fsync", "fdatasync",
        "mmap", "mmap2", "mprotect", "munmap", "mremap", "madvise", "brk",
        "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "rt_sigsuspend", "rt_sigtimedwait", "sigaltstack",
        "getpid", "getppid", "gettid", "getuid", "geteuid", "getgid", "getegid", "getgroups",
        "clock_gettime", "clock_getres", "clock_nanosleep", "gettimeofday", "time", "nanosleep",
        "futex", "futex_time64", "set_tid_address", "set_robust_list", "rseq", "arch_prctl",
        "sched_getaffinity", "sched_yield", "getrandom", "getrlimit", "setrlimit", "getrusage", "umask",
        "dup", "dup2", "dup3", "pipe", "pipe2", "poll", "ppoll", "select", "pselect6",
        "epoll_create", "epoll_create1", "epoll_ctl", "epoll_wait", "epoll_pwait", "epoll_pwait2",
        "capget", "capset", "prctl", "landlock_create_ruleset", "landlock_add_rule", "landlock_restrict_self",
        "execve", "exit", "exit_group", "uname", "sysinfo", "restart_syscall"
    };
    const int commands[] = {F_DUPFD, F_DUPFD_CLOEXEC, F_GETFD, F_SETFD, F_GETFL, F_GETLK, F_SETLK, F_SETLKW};
    if (argc != 2) {
        fputs("Usage: python-policy OUTPUT\n", stderr);
        return 1;
    }
    scmp_filter_ctx policy = seccomp_init(SCMP_ACT_ERRNO(EPERM));
    if (!policy)
        return 1;
    int result = 0;
    for (size_t i = 0; i < sizeof(calls) / sizeof(calls[0]); i++)
        result |= allow(policy, calls[i]);
    result |= seccomp_rule_add(policy, SCMP_ACT_ALLOW, SCMP_SYS(prlimit64), 1, SCMP_A0(SCMP_CMP_EQ, 0));
    for (size_t i = 0; i < sizeof(commands) / sizeof(commands[0]); i++)
        result |= seccomp_rule_add(policy, SCMP_ACT_ALLOW, SCMP_SYS(fcntl), 1, SCMP_A1(SCMP_CMP_EQ, commands[i]));
    result |= seccomp_rule_add(policy, SCMP_ACT_ALLOW, SCMP_SYS(fcntl), 2,
                              SCMP_A1(SCMP_CMP_EQ, F_SETFL), SCMP_A2(SCMP_CMP_MASKED_EQ, O_ASYNC, 0));
    result |= seccomp_rule_add(policy, SCMP_ACT_ALLOW, SCMP_SYS(ioctl), 1, SCMP_A1(SCMP_CMP_EQ, FIOCLEX));
    result |= seccomp_rule_add(policy, SCMP_ACT_ALLOW, SCMP_SYS(ioctl), 1, SCMP_A1(SCMP_CMP_EQ, FIONCLEX));
    int output = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC, 0444);
    if (output < 0)
        result = -1;
    if (!result)
        result = seccomp_export_bpf(policy, output);
    if (output >= 0)
        close(output);
    seccomp_release(policy);
    if (result)
        fputs("Could not build the Python syscall policy\n", stderr);
    return result ? 1 : 0;
}
