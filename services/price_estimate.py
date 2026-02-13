"""Predictive pricing: estimate fair market value and time-to-sale from historical data.

Hybrid search: (1) Vector search (Pinecone) for comparable sales → fetch full details
from PostgreSQL; (2) If no/few results, fall back to PostgreSQL keyword search.

Uses Hye Aero's sale history (aircraft_sales). Keyword path matches model by
manufacturer+model and region mapping (Europe = UK, England, etc.).
"""

import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from database.postgres_client import PostgresClient

if TYPE_CHECKING:
    from rag.embedding_service import EmbeddingService
    from vector_store.pinecone_client import PineconeClient

logger = logging.getLogger(__name__)

# Minimum vector similarity (cosine) to treat a sale as a comparable
VECTOR_SCORE_THRESHOLD = 0.45

# Same region mapping as Market Comparison: e.g. Europe includes UK, England, London
REGION_SEARCH_TERMS: Dict[str, List[str]] = {
    "europe": [
        "europe", "eu", "uk", "united kingdom", "england", "scotland", "wales", "london",
        "france", "germany", "spain", "italy", "netherlands", "belgium", "switzerland",
        "austria", "ireland", "portugal", "greece", "poland", "sweden", "norway", "denmark",
        "finland", "czech", "hungary", "romania", "bulgaria", "croatia", "slovakia", "slovenia",
        "luxembourg", "malta", "cyprus", "estonia", "latvia", "lithuania", "iceland",
    ],
    "north america": [
        "north america", "usa", "united states", "u.s.", "us ", " u.s ", "america",
        "canada", "mexico", "california", "texas", "florida", "new york", "nevada", "arizona",
        "georgia", "illinois", "ohio", "colorado", "washington", "ontario", "quebec",
    ],
    "asia pacific": [
        "asia", "pacific", "australia", "japan", "china", "singapore", "hong kong",
        "uae", "dubai", "india", "south korea", "new zealand", "thailand", "malaysia",
        "indonesia", "philippines", "vietnam", "taiwan",
    ],
}


def _build_vector_query(
    manufacturer: Optional[str] = None,
    model: Optional[str] = None,
    region: Optional[str] = None,
    year: Optional[int] = None,
) -> str:
    """Build a natural-language query for vector search over aircraft sales."""
    parts = ["Aircraft sale", "sold price", "valuation"]
    if manufacturer or model:
        parts.append((manufacturer or "") + " " + (model or ""))
    if region and (region.lower().strip() != "global"):
        parts.append(region)
    if year is not None:
        parts.append(str(year))
    return " ".join(p for p in parts if p.strip())


def _valuation_from_sales(
    sales: List[Dict[str, Any]],
    flight_hours: Optional[float] = None,
    source_note: str = "comparable sale(s)",
) -> Dict[str, Any]:
    """Compute valuation response from a list of sale rows (from vector or keyword)."""
    if not sales:
        return {
            "estimated_value_millions": None,
            "range_low_millions": None,
            "range_high_millions": None,
            "confidence_pct": 0,
            "market_demand": "Unknown",
            "vs_average_pct": None,
            "time_to_sale_days": None,
            "breakdown": [],
            "error": None,
            "message": "No comparable sales found in database. Add more historical data for accurate estimates.",
        }
    prices = [float(s["sold_price"]) for s in sales if s.get("sold_price")]
    if not prices:
        return {
            "estimated_value_millions": None,
            "range_low_millions": None,
            "range_high_millions": None,
            "confidence_pct": 0,
            "market_demand": "Unknown",
            "vs_average_pct": None,
            "time_to_sale_days": None,
            "breakdown": [],
            "error": None,
            "message": "No sale prices in comparables.",
        }
    avg_price = sum(prices) / len(prices)
    low = min(prices)
    high = max(prices)
    if flight_hours is not None and flight_hours > 0:
        avg_hrs = sum(float(s.get("airframe_total_time") or 0) for s in sales) / max(len(sales), 1)
        if avg_hrs > 0:
            hr_factor = 1 - 0.02 * ((flight_hours - avg_hrs) / 1000)
            hr_factor = max(0.7, min(1.2, hr_factor))
            avg_price *= hr_factor
            low *= 0.95
            high *= 1.05
    days_list = [s.get("days_on_market") for s in sales if s.get("days_on_market") is not None]
    avg_days = int(sum(days_list) / len(days_list)) if days_list else None
    return {
        "estimated_value_millions": round(avg_price / 1_000_000, 1),
        "range_low_millions": round(low / 1_000_000, 1),
        "range_high_millions": round(high / 1_000_000, 1),
        "confidence_pct": min(95, 70 + len(prices)),
        "market_demand": "High" if len(prices) >= 10 else "Moderate" if len(prices) >= 3 else "Low",
        "vs_average_pct": None,
        "time_to_sale_days": avg_days,
        "breakdown": [
            {"label": "Base (comparable sales)", "value_millions": round(avg_price / 1_000_000, 2)},
        ],
        "error": None,
        "message": f"Based on {len(prices)} {source_note}.",
    }


def _sales_from_vector_search(
    db: PostgresClient,
    embedding_service: "EmbeddingService",
    pinecone_client: "PineconeClient",
    query_text: str,
    region: Optional[str] = None,
    year: Optional[int] = None,
    top_k: int = 50,
    score_threshold: float = VECTOR_SCORE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Run vector search for aircraft_sale entities, then fetch full rows from PostgreSQL.
    Returns list of sale dicts with sold_price; empty if no matches or Pinecone/embed fails.
    """
    vector = embedding_service.embed_text(query_text)
    if not vector:
        logger.debug("Vector search skip: embed failed")
        return []
    if not pinecone_client.index:
        logger.debug("Vector search skip: Pinecone not connected")
        return []
    try:
        matches = pinecone_client.query(
            vector=vector,
            top_k=top_k,
            filter={"entity_type": "aircraft_sale"},
        )
    except Exception as e:
        logger.warning("Pinecone query failed for price estimate: %s", e)
        return []
    # Pinecone SDK may return matches with .metadata or as dict
    def get_meta(m: Any) -> dict:
        if hasattr(m, "metadata"):
            return getattr(m, "metadata") or {}
        return m.get("metadata") or {}

    seen_ids: set = set()
    sale_ids: List[str] = []
    for m in matches:
        meta = get_meta(m)
        score = getattr(m, "score", None) if hasattr(m, "score") else m.get("score")
        if score is not None and score < score_threshold:
            continue
        eid = (meta.get("entity_id") or "").strip()
        if not eid or eid in seen_ids:
            continue
        seen_ids.add(eid)
        sale_ids.append(eid)
    if not sale_ids:
        return []
    # Fetch full rows from PostgreSQL (detailed data)
    placeholders = ",".join(["%s"] * len(sale_ids))
    query = f"""
        SELECT id, sold_price, ask_price, days_on_market, manufacturer_year, airframe_total_time,
               based_country, registration_country, manufacturer, model
        FROM aircraft_sales
        WHERE id::text IN ({placeholders})
        AND sold_price IS NOT NULL AND sold_price > 0
    """
    rows = db.execute_query(query, tuple(sale_ids))
    if not rows:
        return []
    # Optional server-side region filter (same as keyword path)
    region_lower = (region or "").lower().strip()
    if region_lower and region_lower != "global":
        terms = REGION_SEARCH_TERMS.get(region_lower)
        if terms:
            def in_region(r: Dict[str, Any]) -> bool:
                bc = (r.get("based_country") or "").lower()
                rc = (r.get("registration_country") or "").lower()
                return any(t in bc or t in rc for t in terms)
            rows = [r for r in rows if in_region(r)]
    if year is not None:
        rows = [r for r in rows if r.get("manufacturer_year") == year]
    # Keep order by relevance (order from Pinecone)
    id_order = {eid: i for i, eid in enumerate(sale_ids)}
    rows.sort(key=lambda r: id_order.get(str(r.get("id")), 999))
    return rows[:50]


def estimate_value_hybrid(
    db: PostgresClient,
    embedding_service: Optional["EmbeddingService"],
    pinecone_client: Optional["PineconeClient"],
    manufacturer: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
    flight_hours: Optional[float] = None,
    flight_cycles: Optional[int] = None,
    region: Optional[str] = None,
    vector_score_threshold: float = VECTOR_SCORE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Try vector search first; if enough comparables, return valuation from Postgres details.
    Otherwise fall back to PostgreSQL keyword search.
    """
    query_text = _build_vector_query(manufacturer=manufacturer, model=model, region=region, year=year)
    if embedding_service and pinecone_client and pinecone_client.index:
        sales = _sales_from_vector_search(
            db=db,
            embedding_service=embedding_service,
            pinecone_client=pinecone_client,
            query_text=query_text,
            region=region,
            year=year,
            top_k=50,
            score_threshold=vector_score_threshold,
        )
        if sales:
            out = _valuation_from_sales(sales, flight_hours=flight_hours, source_note="comparable sale(s) (vector search)")
            return out
        logger.info("Price estimate: vector search returned no/few comparables, falling back to keyword search")
    return estimate_value(
        db=db,
        manufacturer=manufacturer,
        model=model,
        year=year,
        flight_hours=flight_hours,
        flight_cycles=flight_cycles,
        region=region,
    )


def estimate_value(
    db: PostgresClient,
    manufacturer: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
    flight_hours: Optional[float] = None,
    flight_cycles: Optional[int] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return estimated market value, range, confidence, and time-to-sale placeholder.
    """
    try:
        conditions = ["1=1"]
        params: list = []
        if manufacturer:
            conditions.append("manufacturer ILIKE %s")
            params.append(f"%{manufacturer}%")
        if model:
            # Match: model column, "manufacturer model" combined, or first-word=manufacturer + rest=model
            # so "BEECH P35" matches Beech+Bonanza P35, Beech+P35, and "BEECH 300" matches Beech+300
            model_conditions = [
                "model ILIKE %s",
                "(COALESCE(manufacturer,'') || ' ' || COALESCE(model,'')) ILIKE %s",
            ]
            model_params: list = [f"%{model}%", f"%{model}%"]
            parts = model.strip().split(None, 1)  # first word, rest
            if len(parts) >= 2:
                first_word, rest = parts[0], parts[1]
                model_conditions.append("(manufacturer ILIKE %s AND model ILIKE %s)")
                model_params.extend([f"%{first_word}%", f"%{rest}%"])
                # Also match P-35 when user types P35 (common variant)
                if rest.replace("-", "") != rest and len(rest) >= 2:
                    alt = rest.replace("-", "")
                    if alt not in (p for i, p in enumerate(model_params) if i % 2 == 1):
                        model_conditions.append("(manufacturer ILIKE %s AND model ILIKE %s)")
                        model_params.extend([f"%{first_word}%", f"%{alt}%"])
                elif len(rest) >= 2 and rest[0].isalpha() and any(c.isdigit() for c in rest) and "-" not in rest:
                    alt = rest[0] + "-" + rest[1:] if len(rest) > 1 else rest
                    model_conditions.append("(manufacturer ILIKE %s AND model ILIKE %s)")
                    model_params.extend([f"%{first_word}%", f"%{alt}%"])
            conditions.append("(" + " OR ".join(model_conditions) + ")")
            params.extend(model_params)
        if year is not None:
            conditions.append("manufacturer_year = %s")
            params.append(year)
        if region and region.lower().strip() != "global":
            terms = REGION_SEARCH_TERMS.get(region.lower().strip())
            if terms:
                placeholders = []
                for _ in terms:
                    placeholders.append("(based_country ILIKE %s OR registration_country ILIKE %s)")
                conditions.append("(" + " OR ".join(placeholders) + ")")
                for t in terms:
                    r = f"%{t}%"
                    params.extend([r, r])
            else:
                conditions.append("(based_country ILIKE %s OR registration_country ILIKE %s)")
                r = f"%{region}%"
                params.extend([r, r])

        where = " AND ".join(conditions)
        params.append(50)

        # Recent sales for same model/year band
        sales_query = f"""
            SELECT sold_price, ask_price, days_on_market, manufacturer_year, airframe_total_time
            FROM aircraft_sales
            WHERE {where}
            AND sold_price IS NOT NULL AND sold_price > 0
            ORDER BY date_sold DESC NULLS LAST
            LIMIT %s
        """
        sales = db.execute_query(sales_query, tuple(params))
        return _valuation_from_sales(sales, flight_hours=flight_hours, source_note="comparable sale(s)")
    except Exception as e:
        logger.exception("Price estimate failed")
        return {
            "estimated_value_millions": None,
            "range_low_millions": None,
            "range_high_millions": None,
            "confidence_pct": 0,
            "market_demand": "Unknown",
            "vs_average_pct": None,
            "time_to_sale_days": None,
            "breakdown": [],
            "error": str(e),
            "message": None,
        }
