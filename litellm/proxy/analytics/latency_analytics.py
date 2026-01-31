
from fastapi import APIRouter, HTTPException, Request, status
from litellm.proxy.utils import PrismaClient, verbose_proxy_logger
from typing import Optional, List
from pydantic import BaseModel
import datetime

router = APIRouter()

class LatencyPercentile(BaseModel):
    model: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    request_count: int

class LatencyAnalyticsResponse(BaseModel):
    latency_percentiles: List[LatencyPercentile]
    start_date: datetime.datetime
    end_date: datetime.datetime
    total_models: int

@router.get("/analytics/latency", response_model=LatencyAnalyticsResponse)
async def get_latency_analytics(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model: Optional[str] = None,
):
    """
    Get P50, P95, P99 latency percentiles grouped by model.
    """
    try:
        prisma_client: Optional[PrismaClient] = request.app.state.prisma_client
        if prisma_client is None:
            raise HTTPException(
                status_code=500, detail="Database connection not initialized"
            )

        # Parse dates (default to last 24h if not provided)
        now = datetime.datetime.now(datetime.timezone.utc)
        if end_date:
            end_dt = datetime.datetime.fromisoformat(end_date)
        else:
            end_dt = now
            
        if start_date:
            start_dt = datetime.datetime.fromisoformat(start_date)
        else:
            start_dt = now - datetime.timedelta(hours=24)

        # Construct Query
        # Note: standard SQL percentile functions might vary by DB (Postgres uses PERCENTILE_CONT)
        # For broader compatibility (including SQLite), we might fetch raw data or use raw query carefully
        # Given this is "Big Data", let's assume Postgres for advanced features, but fallback safe?
        # Let's try to do it in-memory for now if volume is manageable, OR use specific Postgres raw query if detected.
        # Actually, let's keep it simple: Fetch response_time_ms and model for the range.
        
        # However, fetching ALL logs is bad.
        # Let's use a raw query for efficiency.
        
        # Postgres Syntax:
        # SELECT model, 
        #        PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY duration_ms) as p50,
        #        PERCENTILE_CONT(0.95) WITHIN GROUP(ORDER BY duration_ms) as p95, ...
        # FROM "LiteLLM_SpendLogs" ...
        
        sql_query = """
            SELECT 
                model,
                COUNT(*) as request_count,
                PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY (extract(epoch from "endTime") - extract(epoch from "startTime")) * 1000) as p50_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP(ORDER BY (extract(epoch from "endTime") - extract(epoch from "startTime")) * 1000) as p95_ms,
                PERCENTILE_CONT(0.99) WITHIN GROUP(ORDER BY (extract(epoch from "endTime") - extract(epoch from "startTime")) * 1000) as p99_ms
            FROM "LiteLLM_SpendLogs"
            WHERE "startTime" >= $1 AND "startTime" <= $2
        """
        params = [start_dt, end_dt]
        
        if model:
            sql_query += ' AND "model" = $3'
            params.append(model)
            
        sql_query += ' GROUP BY model'

        # Execute
        # We need to handle the case where we are not on Postgres (e.g. SQLite for testing)
        # fallback to fetching data? 
        # For now, let's assume the user is using Postgres as per "1M+ Log Scalability" project context.
        
        results = await prisma_client.db.query_raw(sql_query, *params)
        
        analytics_data = []
        for row in results:
            # Model might be None in DB
            row_model = row.get('model', 'unknown') or 'unknown'
            analytics_data.append(LatencyPercentile(
                model=row_model,
                p50_ms=float(row['p50_ms'] or 0),
                p95_ms=float(row['p95_ms'] or 0),
                p99_ms=float(row['p99_ms'] or 0),
                request_count=int(row['request_count'])
            ))

        return LatencyAnalyticsResponse(
            latency_percentiles=analytics_data,
            start_date=start_dt,
            end_date=end_dt,
            total_models=len(analytics_data)
        )
            
    except Exception as e:
        verbose_proxy_logger.error(f"Error in latency analytics: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {str(e)}"
        )
