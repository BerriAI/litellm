#!/usr/bin/env python3
"""
LiteLLM Simple Working Metrics Exporter
Fixed version that works with the actual database schema
"""

import os
import time
import logging
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from prometheus_client import start_http_server, Gauge, Counter
from prometheus_client.core import CollectorRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database settings
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5433')
DB_USER = os.getenv('DB_USER', 'llmproxy')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'dbpassword9090')
DB_NAME = os.getenv('DB_NAME', 'litellm')

# Exporter settings
METRICS_PORT = int(os.getenv('METRICS_PORT', '9090'))
SCRAPE_INTERVAL = int(os.getenv('SCRAPE_INTERVAL', '60'))

# Create custom registry
registry = CollectorRegistry()

# All metrics as Counters for proper Grafana increase() support
litellm_spend_usd = Counter(
    'litellm_spend_usd_total',
    'LiteLLM spending in USD (cumulative)',
    ['team_id', 'team_alias', 'end_user_id', 'end_user_alias', 'model', 'provider'],
    registry=registry
)

litellm_requests_total = Counter(
    'litellm_requests_total', 
    'Total LiteLLM requests (cumulative)',
    ['team_id', 'team_alias', 'end_user_id', 'end_user_alias', 'model', 'provider', 'status'],
    registry=registry
)

litellm_tokens_total = Counter(
    'litellm_tokens_total',
    'Total tokens processed (cumulative)', 
    ['team_id', 'team_alias', 'end_user_id', 'end_user_alias', 'model', 'provider', 'token_type'],
    registry=registry
)

# Time-based metrics for heatmaps
litellm_requests_by_time_total = Gauge(
    'litellm_requests_by_time_total',
    'Requests aggregated by time patterns',
    ['team_id', 'team_alias', 'end_user_id', 'end_user_alias', 'hour_of_day', 'day_name'],
    registry=registry
)

# Performance metrics
litellm_request_duration_seconds = Gauge(
    'litellm_request_duration_seconds',
    'Average request duration in seconds',
    ['team_id', 'team_alias', 'model', 'provider'],
    registry=registry
)

litellm_tokens_per_second = Gauge(
    'litellm_tokens_per_second',
    'Token generation speed (tokens/second)',
    ['team_id', 'team_alias', 'model', 'provider'],
    registry=registry
)

# Budget and financial metrics
litellm_team_budget_usd = Gauge(
    'litellm_team_budget_usd',
    'Team budget status and usage',
    ['team_id', 'team_alias', 'metric_type'],  # metric_type: max_budget, current_spend, remaining, usage_percent
    registry=registry
)

litellm_cost_efficiency = Gauge(
    'litellm_cost_efficiency',
    'Cost efficiency metrics (cost per token, tokens per dollar)',
    ['team_id', 'team_alias', 'end_user_id', 'end_user_alias', 'model', 'provider', 'metric_type'],  # metric_type: cost_per_token, tokens_per_dollar
    registry=registry
)

class LiteLLMMetricsExporter:
    def __init__(self):
        self.connection = None
        self.last_export_time = datetime.now() - timedelta(hours=1)  # Start from 1 hour ago
        self.connect_to_db()

    def connect_to_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                cursor_factory=RealDictCursor
            )
            logger.info("Connected to LiteLLM database")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def execute_query(self, query, params=None):
        """Execute query with error handling"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query failed: {e}")
            logger.error(f"Query: {query}")
            self.connection.rollback()
            return []

    def export_core_metrics(self):
        """Export new records to Counter metrics since last export"""
        logger.info("Exporting counter metrics...")
        
        # Query for new records since last export
        query = '''
        SELECT 
            COALESCE(sl.team_id, 'no_team') as team_id,
            COALESCE(tt.team_alias, 'no_alias') as team_alias,
            COALESCE(sl.end_user, 'anonymous') as end_user_id,
            COALESCE(eu.alias, COALESCE(sl.end_user, 'anonymous')) as end_user_alias,
            COALESCE(sl.model, 'unknown') as model,
            COALESCE(sl.custom_llm_provider, 'unknown') as provider,
            COALESCE(sl.status, 'unknown') as status,
            sl.spend,
            sl.total_tokens,
            sl.prompt_tokens, 
            sl.completion_tokens
        FROM "LiteLLM_SpendLogs" sl
        LEFT JOIN "LiteLLM_TeamTable" tt ON sl.team_id = tt.team_id
        LEFT JOIN "LiteLLM_EndUserTable" eu ON sl.end_user = eu.user_id
        WHERE sl."startTime" >= %s
        '''
        
        results = self.execute_query(query, [self.last_export_time])
        logger.info(f"Found {len(results)} new records since {self.last_export_time}")
        
        # Increment counters with new data
        for row in results:
            labels = {
                'team_id': str(row['team_id']),
                'team_alias': str(row['team_alias']), 
                'end_user_id': str(row['end_user_id']),
                'end_user_alias': str(row['end_user_alias']),
                'model': str(row['model']),
                'provider': str(row['provider'])
            }
            
            # Increment spend counter
            if row['spend'] and row['spend'] > 0:
                litellm_spend_usd.labels(**labels).inc(float(row['spend']))
            
            # Increment request counter
            request_labels = labels.copy()
            request_labels['status'] = str(row['status'])
            litellm_requests_total.labels(**request_labels).inc(1)
            
            # Increment token counters
            if row['total_tokens'] and row['total_tokens'] > 0:
                litellm_tokens_total.labels(**labels, token_type='total').inc(float(row['total_tokens']))
            if row['prompt_tokens'] and row['prompt_tokens'] > 0:
                litellm_tokens_total.labels(**labels, token_type='prompt').inc(float(row['prompt_tokens']))
            if row['completion_tokens'] and row['completion_tokens'] > 0:
                litellm_tokens_total.labels(**labels, token_type='completion').inc(float(row['completion_tokens']))
        
        # Update last export time
        self.last_export_time = datetime.now()

    def export_time_patterns(self):
        """Export time-based usage patterns"""
        logger.info("Exporting time patterns...")
        
        litellm_requests_by_time_total._metrics.clear()
        
        query = '''
        SELECT 
            COALESCE(sl.team_id, 'no_team') as team_id,
            COALESCE(tt.team_alias, 'no_alias') as team_alias,
            COALESCE(sl.end_user, 'anonymous') as end_user_id,
            COALESCE(eu.alias, COALESCE(sl.end_user, 'anonymous')) as end_user_alias,
            EXTRACT(HOUR FROM sl."startTime" AT TIME ZONE 'UTC' AT TIME ZONE 'Europe/Moscow') as hour_of_day,
            TO_CHAR(sl."startTime" AT TIME ZONE 'UTC' AT TIME ZONE 'Europe/Moscow', 'Day') as day_name,
            COUNT(*) as request_count
        FROM "LiteLLM_SpendLogs" sl
        LEFT JOIN "LiteLLM_TeamTable" tt ON sl.team_id = tt.team_id
        LEFT JOIN "LiteLLM_EndUserTable" eu ON sl.end_user = eu.user_id
        WHERE sl."startTime" >= NOW() - INTERVAL '7 days'
        GROUP BY sl.team_id, tt.team_alias, sl.end_user, eu.alias, 
                 EXTRACT(HOUR FROM sl."startTime"), TO_CHAR(sl."startTime", 'Day')
        HAVING COUNT(*) > 0
        '''
        
        results = self.execute_query(query)
        logger.info(f"Found {len(results)} time pattern records")
        
        for row in results:
            litellm_requests_by_time_total.labels(
                team_id=str(row['team_id']),
                team_alias=str(row['team_alias']),
                end_user_id=str(row['end_user_id']),
                end_user_alias=str(row['end_user_alias']),
                hour_of_day=str(int(row['hour_of_day'])),
                day_name=str(row['day_name']).strip()
            ).set(float(row['request_count']))

    def export_performance_metrics(self):
        """Export performance metrics"""
        logger.info("Exporting performance metrics...")
        
        litellm_request_duration_seconds._metrics.clear()
        litellm_tokens_per_second._metrics.clear()
        
        query = '''
        SELECT 
            COALESCE(sl.team_id, 'no_team') as team_id,
            COALESCE(tt.team_alias, 'no_alias') as team_alias,
            COALESCE(sl.model, 'unknown') as model,
            COALESCE(sl.custom_llm_provider, 'unknown') as provider,
            COUNT(*) as request_count,
            COALESCE(AVG(EXTRACT(EPOCH FROM (sl."endTime" - sl."startTime"))), 0) as avg_duration_seconds,
            COALESCE(SUM(sl.total_tokens), 0) as total_tokens,
            COALESCE(SUM(EXTRACT(EPOCH FROM (sl."endTime" - sl."startTime"))), 0) as total_duration_seconds
        FROM "LiteLLM_SpendLogs" sl
        LEFT JOIN "LiteLLM_TeamTable" tt ON sl.team_id = tt.team_id
        WHERE sl."startTime" >= NOW() - INTERVAL '1 hour'
            AND sl."endTime" IS NOT NULL 
            AND sl."startTime" IS NOT NULL
            AND sl.total_tokens > 0
        GROUP BY sl.team_id, tt.team_alias, sl.model, sl.custom_llm_provider
        HAVING COUNT(*) >= 3  -- Minimum requests for reliable averages
        '''
        
        results = self.execute_query(query)
        logger.info(f"Found {len(results)} performance records")
        
        for row in results:
            labels = {
                'team_id': str(row['team_id']),
                'team_alias': str(row['team_alias']),
                'model': str(row['model']),
                'provider': str(row['provider'])
            }
            
            # Average duration
            litellm_request_duration_seconds.labels(**labels).set(float(row['avg_duration_seconds']))
            
            # Tokens per second
            if row['total_duration_seconds'] > 0:
                tokens_per_sec = row['total_tokens'] / row['total_duration_seconds']
                litellm_tokens_per_second.labels(**labels).set(float(tokens_per_sec))

    def export_team_budget_metrics(self):
        """Export team budget status metrics"""
        logger.info("Exporting team budget metrics...")
        
        litellm_team_budget_usd._metrics.clear()
        
        query = '''
        SELECT 
            team_id,
            COALESCE(team_alias, 'no_alias') as team_alias,
            COALESCE(max_budget, 0) as max_budget,
            COALESCE(spend, 0) as current_spend
        FROM "LiteLLM_TeamTable"
        WHERE team_id IS NOT NULL
        '''
        
        results = self.execute_query(query)
        logger.info(f"Found {len(results)} team budget records")
        
        for row in results:
            team_id = str(row['team_id'])
            team_alias = str(row['team_alias'])
            max_budget = float(row['max_budget'])
            current_spend = float(row['current_spend'])
            
            # Set budget metrics
            litellm_team_budget_usd.labels(team_id=team_id, team_alias=team_alias, metric_type='max_budget').set(max_budget)
            litellm_team_budget_usd.labels(team_id=team_id, team_alias=team_alias, metric_type='current_spend').set(current_spend)
            
            if max_budget > 0:
                remaining = max(0, max_budget - current_spend)
                usage_percent = min(100, (current_spend / max_budget) * 100)
                
                litellm_team_budget_usd.labels(team_id=team_id, team_alias=team_alias, metric_type='remaining').set(remaining)
                litellm_team_budget_usd.labels(team_id=team_id, team_alias=team_alias, metric_type='usage_percent').set(usage_percent)

    def export_cost_efficiency_metrics(self):
        """Export cost efficiency metrics"""
        logger.info("Exporting cost efficiency metrics...")
        
        litellm_cost_efficiency._metrics.clear()
        
        query = '''
        SELECT 
            COALESCE(sl.team_id, 'no_team') as team_id,
            COALESCE(tt.team_alias, 'no_alias') as team_alias,
            COALESCE(sl.end_user, 'anonymous') as end_user_id,
            COALESCE(eu.alias, COALESCE(sl.end_user, 'anonymous')) as end_user_alias,
            COALESCE(sl.model, 'unknown') as model,
            COALESCE(sl.custom_llm_provider, 'unknown') as provider,
            COALESCE(SUM(sl.spend), 0) as total_spend,
            COALESCE(SUM(sl.total_tokens), 0) as total_tokens,
            COUNT(*) as request_count
        FROM "LiteLLM_SpendLogs" sl
        LEFT JOIN "LiteLLM_TeamTable" tt ON sl.team_id = tt.team_id
        LEFT JOIN "LiteLLM_EndUserTable" eu ON sl.end_user = eu.user_id
        WHERE sl."startTime" >= NOW() - INTERVAL '24 hours'
            AND sl.spend > 0 
            AND sl.total_tokens > 0
        GROUP BY sl.team_id, tt.team_alias, sl.end_user, eu.alias, sl.model, sl.custom_llm_provider
        HAVING SUM(sl.spend) > 0 AND SUM(sl.total_tokens) > 0
        '''
        
        results = self.execute_query(query)
        logger.info(f"Found {len(results)} cost efficiency records")
        
        for row in results:
            labels = {
                'team_id': str(row['team_id']),
                'team_alias': str(row['team_alias']),
                'end_user_id': str(row['end_user_id']),
                'end_user_alias': str(row['end_user_alias']),
                'model': str(row['model']),
                'provider': str(row['provider'])
            }
            
            total_spend = float(row['total_spend'])
            total_tokens = float(row['total_tokens'])
            
            if total_tokens > 0:
                # Cost per token (USD per token)
                cost_per_token = total_spend / total_tokens
                litellm_cost_efficiency.labels(**labels, metric_type='cost_per_token').set(cost_per_token)
                
                # Tokens per dollar
                if total_spend > 0:
                    tokens_per_dollar = total_tokens / total_spend
                    litellm_cost_efficiency.labels(**labels, metric_type='tokens_per_dollar').set(tokens_per_dollar)

    def export_all_metrics(self):
        """Export all metrics"""
        start_time = time.time()
        logger.info("=== Starting metrics export ===")
        
        try:
            # Export main Counter metrics (for Grafana increase() functions)
            self.export_core_metrics()
            
            # Export additional metrics
            self.export_time_patterns()
            self.export_performance_metrics()
            self.export_team_budget_metrics()
            self.export_cost_efficiency_metrics()
            
            export_time = time.time() - start_time
            logger.info(f"=== ALL METRICS EXPORTED in {export_time:.2f}s ===")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            # Re-establish connection if needed
            try:
                self.connect_to_db()
            except:
                pass

    def run(self):
        """Main loop"""
        logger.info("============================================================")
        logger.info("🚀 STARTING SIMPLE WORKING LiteLLM METRICS EXPORTER")
        logger.info(f"📊 Metrics server: http://localhost:{METRICS_PORT}/metrics")
        logger.info(f"⏱️  Scrape interval: {SCRAPE_INTERVAL}s")
        logger.info("============================================================")
        
        # Start HTTP server
        start_http_server(METRICS_PORT, registry=registry)
        
        while True:
            self.export_all_metrics()
            time.sleep(SCRAPE_INTERVAL)

if __name__ == "__main__":
    exporter = LiteLLMMetricsExporter()
    exporter.run()