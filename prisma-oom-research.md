# Prisma Python Client OOM / Memory Investigation

Research conducted: 2026-02-26

## LiteLLM's Current Prisma Setup

- **prisma-client-py version**: 0.11.0 (pinned in pyproject.toml)
- **Engine type**: binary (default; no `engineType` override in schema.prisma)
- **Engine version**: ~5.4.2 (compatible with prisma-client-py 0.11.0)
- **Binary targets**: native, debian-openssl-1.1.x, debian-openssl-3.0.x, linux-musl, linux-musl-openssl-3.0.x
- **Schema size**: 1000+ lines, many models with JSON/JSONB fields

## 1. prisma-client-py Specific Memory Issues

### Large Schema Import Memory (prisma-client-py#1040)
- **URL**: https://github.com/RobertCraigie/prisma-client-py/issues/1040
- Large schemas (132+ tables) produce a ~1 million line `types.py` file
- Importing the Prisma client takes **10-20 seconds** and consumes **~1GB of RAM**
- Caused by frequent `model_rebuild()` calls in Pydantic
- Particularly problematic for serverless / memory-constrained environments
- **Applies to LiteLLM**: Yes - the schema has many models

## 2. Prisma Query Engine Binary Memory Issues

### a) RSS Memory Growth - Not Released (prisma/prisma#21471)
- **URL**: https://github.com/prisma/prisma/issues/21471
- RSS (Resident Set Size) grows significantly during operations and **is NOT freed**
- Heap memory returns to baseline, but RSS stays elevated until process restart
- Particularly bad with JSONB fields (20-500KB each) across hundreds/thousands of records
- Root cause: JSON marshalling in the Rust engine
- **Applies to LiteLLM**: Yes - many tables store JSON metadata

### b) Large Query Memory Explosion (prisma/prisma#22387)
- **URL**: https://github.com/prisma/prisma/issues/22387
- Querying 1M+ records: "Failed to convert rust String into napi string"
- Memory scales: 100K records → 0.4GB, 2M records → 7.3GB, 3M records → 11.1GB
- Data marshalling between Rust engine and host runtime is the bottleneck

### c) Memory Leak Across Versions (prisma/prisma#23661)
- **URL**: https://github.com/prisma/prisma/issues/23661
- Prisma 4.2.1 → 5.11.0: memory **doubled from 700MB to 1.5GB**
- Memory increases continuously over time (leak pattern)
- RSS grows under concurrent queries with larger datasets

### d) Binary Engine Process Persistence (prisma/prisma#7020)
- **URL**: https://github.com/prisma/prisma/issues/7020
- Query engine binary **continues running** after parent process is killed
- Creates connection pool leaks - engine keeps connections open
- Leads to exhausted database connections on restart
- **Applies to LiteLLM**: Yes - binary engine mode spawns separate process

### e) Disconnect Does Not Free Memory (prisma/prisma#9044)
- **URL**: https://github.com/prisma/prisma/issues/9044
- `$disconnect()` does not properly free memory or kill the query engine
- Creating/disconnecting multiple PrismaClient instances: memory grows continuously
- Testing: **1605.77 MB** after 60 min (leaked) vs **288.34 MB** (proper cleanup)

### f) Migration OOM Crashes (prisma/prisma#24061)
- **URL**: https://github.com/prisma/prisma/issues/24061
- Prisma 5.13.0 regression: `prisma migrate deploy` consumes excessive memory
- Containers with 128MB crash; workaround: 512MB or downgrade to 5.12.1

### g) RSS with Concurrent Queries (prisma/prisma#25371)
- **URL**: https://github.com/prisma/prisma/issues/25371
- RSS memory leaks worsen with concurrent queries on larger datasets
- Service crashes even with scaled resources (tested up to 32GB RAM)

### h) Repeated Query Memory Growth (prisma/prisma#18934)
- **URL**: https://github.com/prisma/prisma/issues/18934
- Running the same query 10,000 times: heap grew from 8.7MB to 31.4MB (3.34x)
- Memory eventually stabilizes but at elevated level

### i) Statement Cache Memory (prisma discussions#15971)
- **URL**: https://github.com/prisma/prisma/discussions/15971
- Up to 12GB memory usage linked to statement cache in Docker
- Partially addressed in Prisma 4.0.0 but may still contribute

## 3. Binary Engine vs Library Engine

- **Library engine** leaks approximately **2x faster** than binary engine
- **Binary engine** spawns a separate `query-engine` process that independently accumulates memory
- Neither engine type is immune to memory leaks
- Source: https://errorism.dev/issues/prisma-prisma-prisma-memory-leak-when-using-in-nestjs-app-reported-by-jest

For prisma-client-py: Only binary engine is available. The engine runs as a **separate OS process**, meaning:
- Memory consumption is in a **child process**, not the Python process itself
- The child process can grow unbounded, not subject to Python memory management
- If the Python process crashes, the engine process may persist as an orphan

## 4. Known Fixes and Workarounds

### a) Connection Pool Limits
- Set `connection_limit` in DATABASE_URL: `?connection_limit=5`
- Default is 10 connections; each consumes RAM
- Docs: https://prisma.io/docs/orm/prisma-client/setup-and-configuration/databases-connections/connection-pool

### b) Single Global PrismaClient Instance
- Multiple instances create separate connection pools and engine processes
- Use one global instance and share it
- Docs: https://www.prisma.io/docs/v6/orm/prisma-client/setup-and-configuration/databases-connections/connection-management

### c) Periodic Disconnect/Reconnect (Engine Restart)
- `disconnect()` → `connect()` restarts the query engine process, reclaiming memory
- For prisma-client-py 0.11.0, this should work
- **Caveat**: In Prisma 6.6.0+ Node.js, this broke (prisma/prisma#27157)

### d) Limit Result Set Sizes
- Use `select` to limit returned fields (avoid large JSONB columns)
- Paginate queries instead of fetching millions of records
- Avoid returning large JSON columns when not needed

### e) Reduce Statement Cache Size
- Lower `statement_cache_size` parameter in connection URL
- e.g., `?statement_cache_size=100`

### f) Container Memory Limits + Restart Policy
- Set memory limits to prevent unbounded growth
- Use Kubernetes/Docker restart policies to handle OOM kills
- Accept periodic restarts as normal under this constraint

### g) Monitor Query Engine Process RSS
- Implement a watchdog monitoring the query-engine child process RSS
- Proactively restart the service when memory exceeds a threshold

### h) Upgrade Path (Not Immediately Available)
- Prisma v7 removes the Rust engine entirely (TypeScript-based query compiler)
- Prisma v6.16.0+ supports `engineType = "client"` with driver adapters
- **Not applicable to prisma-client-py 0.11.0** which requires the Rust binary engine

## 5. Recommendations for LiteLLM

1. **Reduce connection_limit** in DATABASE_URL to minimize per-connection memory overhead
2. **Implement periodic Prisma client restart** - disconnect and reconnect on a schedule (e.g., every few hours) to reclaim leaked memory from the query engine process
3. **Audit JSON field queries** - ensure large JSON/JSONB columns aren't fetched unnecessarily (use `select` to limit fields)
4. **Monitor the query-engine process** - track its RSS independently from the Python process
5. **Set container memory limits** with restart policies as a safety net
6. **Consider batch query sizes** - paginate large result sets
7. **Long-term**: Track prisma-client-py updates; newer versions may eventually support driver adapters or the TypeScript query compiler, removing the Rust engine dependency
