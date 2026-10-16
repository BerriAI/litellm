// procmon samples one process's CPU, memory, threads, open files and established TCP connections
// from /proc and writes a JSON line per sample. Run it in the gateway's PID namespace
// (docker run --pid=container:gateway ... procmon -pid 1) so it needs no agent in the gateway.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"
)

const clockTicks = 100 // USER_HZ on Linux

func cpuTicks(pid int) (int64, error) {
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return 0, err
	}
	// Fields after the parenthesised command name; utime and stime are fields 14 and 15.
	rest := string(b[strings.LastIndexByte(string(b), ')')+2:])
	f := strings.Fields(rest)
	u, _ := strconv.ParseInt(f[11], 10, 64)
	s, _ := strconv.ParseInt(f[12], 10, 64)
	return u + s, nil
}

func status(pid int) (rssKB, threads int64) {
	f, err := os.Open(fmt.Sprintf("/proc/%d/status", pid))
	if err != nil {
		return
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if v, ok := strings.CutPrefix(line, "VmRSS:"); ok {
			rssKB, _ = strconv.ParseInt(strings.Fields(v)[0], 10, 64)
		}
		if v, ok := strings.CutPrefix(line, "Threads:"); ok {
			threads, _ = strconv.ParseInt(strings.TrimSpace(v), 10, 64)
		}
	}
	return
}

func countDir(path string) int {
	entries, err := os.ReadDir(path)
	if err != nil {
		return -1
	}
	return len(entries)
}

// established counts ESTABLISHED sockets (state 01) in the process's network namespace.
func established(pid int) int {
	n := 0
	for _, name := range []string{"tcp", "tcp6"} {
		f, err := os.Open(fmt.Sprintf("/proc/%d/net/%s", pid, name))
		if err != nil {
			continue
		}
		sc := bufio.NewScanner(f)
		sc.Scan()
		for sc.Scan() {
			if fields := strings.Fields(sc.Text()); len(fields) > 3 && fields[3] == "01" {
				n++
			}
		}
		f.Close()
	}
	return n
}

func main() {
	pid := flag.Int("pid", 1, "process to watch")
	every := flag.Duration("every", 5*time.Second, "sample interval")
	flag.Parse()
	enc := json.NewEncoder(os.Stdout)
	prev, err := cpuTicks(*pid)
	if err != nil {
		log.Fatal(err)
	}
	prevAt := time.Now()
	for range time.Tick(*every) {
		ticks, err := cpuTicks(*pid)
		if err != nil {
			log.Fatal(err)
		}
		now := time.Now()
		rss, threads := status(*pid)
		enc.Encode(map[string]any{
			"t":         now.UnixMilli(),
			"cpu_cores": float64(ticks-prev) / clockTicks / now.Sub(prevAt).Seconds(),
			"rss_mb":    rss / 1024,
			"threads":   threads,
			"fds":       countDir(fmt.Sprintf("/proc/%d/fd", *pid)),
			"tcp_estab": established(*pid),
		})
		prev, prevAt = ticks, now
	}
}
