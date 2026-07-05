"""
Health check endpoint — system status, pipeline status, DB connection, scheduler state.
"""

from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, func
from datetime import datetime, timezone

from app.database import get_db
from app.config import settings

router = APIRouter()


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Full system health check — DB, API keys, pipeline status, scheduler, data stats."""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }

    # DB connection
    try:
        await db.execute(text("SELECT 1"))
        health_data["checks"]["database"] = {"status": "connected", "ok": True}
    except Exception as e:
        health_data["checks"]["database"] = {"status": "error", "ok": False, "message": str(e)}
        health_data["status"] = "unhealthy"

    # API keys
    health_data["checks"]["api_keys"] = {
        "groq": bool(settings.groq_api_key),
        "reddit": bool(settings.reddit_client_id),
        "twitter": bool(settings.twitter_bearer_token),
    }

    # Data counts
    try:
        from app.models.raw_review import RawReview
        from app.models.classified_review import ClassifiedReview

        raw_count = (await db.execute(select(func.count()).select_from(RawReview))).scalar() or 0
        classified_count = (await db.execute(select(func.count()).select_from(ClassifiedReview))).scalar() or 0
        failed_count = (await db.execute(
            select(func.count()).select_from(ClassifiedReview).where(ClassifiedReview.classification_failed == True)
        )).scalar() or 0

        health_data["checks"]["data"] = {
            "raw_reviews": raw_count,
            "classified_reviews": classified_count,
            "classification_failure_count": failed_count,
            "classification_failure_rate": round(failed_count / max(classified_count, 1) * 100, 2),
        }
    except Exception as e:
        health_data["checks"]["data"] = {"error": str(e)}

    # Scheduler status
    try:
        from app.scheduler import scheduler
        jobs = scheduler.get_jobs()
        health_data["checks"]["scheduler"] = {
            "running": scheduler.running,
            "jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                }
                for job in jobs
            ],
        }
    except Exception as e:
        health_data["checks"]["scheduler"] = {"error": str(e)}

    # Last pipeline run per job type
    try:
        result = await db.execute(text("""
            SELECT DISTINCT ON (job_name)
                job_name, status, entries_processed, completed_at, duration_seconds
            FROM pipeline_runs
            ORDER BY job_name, started_at DESC
        """))
        runs = result.fetchall()
        health_data["checks"]["pipeline_runs"] = [
            {
                "job": r[0],
                "status": r[1],
                "entries": r[2],
                "completed_at": r[3].isoformat() if r[3] else None,
                "duration_s": r[4],
            }
            for r in runs
        ]
    except Exception:
        health_data["checks"]["pipeline_runs"] = []

    return health_data


@router.post("/pipeline/trigger/{job_id}")
async def trigger_job(job_id: str, background_tasks: BackgroundTasks):
    """
    Manually trigger a specific pipeline job.
    job_id: one of [ingest_all_sources, classify_new_reviews, embed_new_reviews,
                    refresh_synthesis_cache, update_clusters]
    """
    from app.scheduler import (
        job_ingest_all_sources,
        job_classify_new_reviews,
        job_embed_new_reviews,
        job_refresh_synthesis_cache,
        job_update_clusters,
    )

    job_map = {
        "ingest_all_sources": job_ingest_all_sources,
        "classify_new_reviews": job_classify_new_reviews,
        "embed_new_reviews": job_embed_new_reviews,
        "refresh_synthesis_cache": job_refresh_synthesis_cache,
        "update_clusters": job_update_clusters,
    }

    if job_id not in job_map:
        return {"error": f"Unknown job_id. Valid: {list(job_map.keys())}"}

    background_tasks.add_task(job_map[job_id])
    return {
        "message": f"Job '{job_id}' triggered in background",
        "triggered_at": datetime.now(timezone.utc).isoformat(),
    }
