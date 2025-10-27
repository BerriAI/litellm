#!/usr/bin/env python3
"""
Historical Data Import Script for LiteLLM Prometheus
Creates time series data from database for proper historical analysis
"""

import os
import psycopg2
import requests
from datetime import datetime, timedelta
from collections import defaultdict

# Database settings
DB_HOST = 'localhost'
DB_PORT = '5433'
DB_USER = 'llmproxy'
DB_PASSWORD = 'dbpassword9090'
DB_NAME = 'litellm'

def get_historical_data():
    """Get daily aggregated data from database"""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, 
        user=DB_USER, password=DB_PASSWORD, 
        database=DB_NAME
    )
    
    query = """
    SELECT 
        DATE("startTime") as date,
        team_id,
        COALESCE(metadata->>'user_api_key_team_alias', 'no_alias') as team_alias,
        end_user,
        COALESCE(metadata->>'user_api_key_end_user_alias', end_user, '') as end_user_alias,
        model,
        custom_llm_provider as provider,
        CASE WHEN status = 'success' THEN 'success' ELSE 'failure' END as status,
        COUNT(*) as request_count,
        SUM(spend) as total_spend,
        SUM(total_tokens) as total_tokens,
        SUM(prompt_tokens) as prompt_tokens,
        SUM(completion_tokens) as completion_tokens
    FROM "LiteLLM_SpendLogs"
    WHERE "startTime" >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY 1,2,3,4,5,6,7,8
    ORDER BY date ASC;
    """
    
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    
    return results

def format_daily_summary(data):
    """Format data into daily summary"""
    daily_stats = defaultdict(lambda: {'requests': 0, 'spend': 0, 'teams': set()})
    
    for row in data:
        date = row[0]
        request_count = row[8]
        spend = row[9] or 0
        team_alias = row[2] or 'no_alias'
        
        daily_stats[date]['requests'] += request_count
        daily_stats[date]['spend'] += spend
        daily_stats[date]['teams'].add(team_alias)
    
    return daily_stats

def main():
    print("🔍 Checking historical data consistency...")
    
    # Get data from database
    print("📊 Querying database...")
    historical_data = get_historical_data()
    print(f"Found {len(historical_data)} data points")
    
    # Format summary
    daily_summary = format_daily_summary(historical_data)
    
    print("\n📅 Daily Summary (Last 10 days):")
    print("Date       | Requests | Spend    | Teams")
    print("-" * 45)
    
    for date in sorted(daily_summary.keys())[-10:]:
        stats = daily_summary[date]
        teams_str = ', '.join(sorted(stats['teams'])[:3])
        if len(stats['teams']) > 3:
            teams_str += f" (+{len(stats['teams'])-3} more)"
        
        print(f"{date} | {stats['requests']:8d} | ${stats['spend']:7.2f} | {teams_str}")
    
    # Check current Prometheus data
    print(f"\n🔍 Current Prometheus data:")
    try:
        response = requests.get('http://localhost:9092/api/v1/query?query=sum(litellm_requests_total)')
        if response.status_code == 200:
            data = response.json()
            if data['data']['result']:
                current_total = int(float(data['data']['result'][0]['value'][1]))
                print(f"Current total in Prometheus: {current_total:,}")
            else:
                print("No data in Prometheus")
        else:
            print("Cannot query Prometheus")
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
    
    # Validation
    total_historical = sum(stats['requests'] for stats in daily_summary.values())
    print(f"\n✅ Historical total from DB: {total_historical:,}")
    
    # Check for gaps
    dates = sorted(daily_summary.keys())
    if dates:
        start_date = dates[0]
        end_date = dates[-1]
        expected_days = (end_date - start_date).days + 1
        actual_days = len(dates)
        
        print(f"📅 Date range: {start_date} to {end_date}")
        print(f"📊 Expected days: {expected_days}, Actual days: {actual_days}")
        
        if expected_days != actual_days:
            print("⚠️  Missing days detected!")
        else:
            print("✅ No missing days")

if __name__ == "__main__":
    main()