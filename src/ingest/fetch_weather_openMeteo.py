
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEZONE = "Australia/Sydney"
DEFAULT_REGIONS = {
    "Sydney East",
    "Sydney South-west",
    "Sydney North-west",
}

# The first four variables directly correspond to AirEquity's weather features.
# The remaining fields are retained as optional future covariates for prediction.
DEFAULT_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "surface_pressure",
    "shortwave_radiation",
]

# Friendly aliases make downstream feature merging clear without losing the
# original Open-Meteo variable names in the normalised Parquet output.
FEATURE_ALIASES = {
    "temperature_2m": "temp_forecast_c",
    "relative_humidity_2m": "humid_forecast_pct",
    "wind_speed_10m": "wsp_forecast_m_s",
    "wind_direction_10m": "wdr_forecast_deg",
}

DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_MAX_RETRIES = 4


def utc_now() -> datetime:
    """Return a timezone-aware current UTC timestamp."""
    return datetime.now(timezone.utc)


def parse_station_argument(value: str) -> dict[str, Any]:
    """Parse ``site_id,name,latitude,longitude[,region]`` from the command line."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in (4, 5):
        raise argparse.ArgumentTypeError(
            "--station must be 'site_id,name,latitude,longitude[,region]'"
        )

    try:
        latitude = float(parts[2])
        longitude = float(parts[3])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "station latitude and longitude must be numeric"
        ) from exc

    return {
        "site_id": parts[0],
        "site_name": parts[1],
        "latitude": latitude,
        "longitude": longitude,
        "region": parts[4] if len(parts) == 5 else None,
    }


def first_present(record: dict[str, Any], *keys: str) -> Any:
    """Return the first non-empty value found under one of ``keys``."""
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def validate_coordinates(latitude: Any, longitude: Any, label: str) -> tuple[float, float]:
    """Coerce and validate WGS84 coordinates for a station."""
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: coordinates are missing or non-numeric") from exc

    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError(f"{label}: coordinates must be finite")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"{label}: coordinates are outside WGS84 bounds")
    return lat, lon


def normalise_site(record: dict[str, Any]) -> dict[str, Any]:
    """Convert NSW API-style or simple site records into one internal schema."""
    site_id = first_present(record, "Site_Id", "site_id", "id")
    name = first_present(record, "SiteName", "site_name", "name")
    region = first_present(record, "Region", "region")
    label = str(name or site_id or "unknown site")
    latitude, longitude = validate_coordinates(
        first_present(record, "Latitude", "latitude", "lat"),
        first_present(record, "Longitude", "longitude", "lon", "lng"),
        label,
    )

    return {
        "site_id": str(site_id) if site_id is not None else None,
        "site_name": str(name) if name is not None else label,
        "region": str(region) if region is not None else None,
        "latitude": latitude,
        "longitude": longitude,
    }


def load_sites(path: Path, include_all_sites: bool) -> tuple[list[dict[str, Any]], list[str]]:
    """Load and filter station records from a JSON site registry."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        payload = payload.get("sites") or payload.get("data") or payload.get("Values")
    if not isinstance(payload, list):
        raise ValueError("sites file must contain a JSON list or an object with a sites/data/Values list")

    sites: list[dict[str, Any]] = []
    skipped: list[str] = []
    seen: set[tuple[str | None, float, float]] = set()

    for raw_record in payload:
        if not isinstance(raw_record, dict):
            skipped.append("non-object site record")
            continue

        try:
            site = normalise_site(raw_record)
        except ValueError as exc:
            skipped.append(str(exc))
            continue

        name = site["site_name"].lower()
        region = site["region"] or ""
        if "test" in name or "test" in region.lower():
            skipped.append(f"{site['site_name']}: test station")
            continue
        if not include_all_sites and region not in DEFAULT_REGIONS:
            skipped.append(f"{site['site_name']}: outside Greater Sydney region filter")
            continue

        identity = (site["site_id"], site["latitude"], site["longitude"])
        if identity not in seen:
            sites.append(site)
            seen.add(identity)

    return sites, skipped


def request_forecast(
    session: requests.Session,
    site: dict[str, Any],
    hourly_variables: list[str],
    forecast_days: int,
    past_days: int,
    timezone_name: str,
    timeout_seconds: int,
    max_retries: int,
) -> dict[str, Any]:
    """Retrieve one station forecast with bounded exponential-backoff retries."""
    params = {
        "latitude": f"{site['latitude']:.6f}",
        "longitude": f"{site['longitude']:.6f}",
        "hourly": ",".join(hourly_variables),
        "timezone": timezone_name,
        "forecast_days": forecast_days,
        "past_days": past_days,
        # NSW wind-speed data is conventionally represented in m/s. Requesting
        # this unit makes future feature comparisons explicit and consistent.
        "wind_speed_unit": "ms",
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                OPEN_METEO_FORECAST_URL,
                params=params,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            validate_forecast_payload(payload, hourly_variables)
            return payload
        except (requests.RequestException, ValueError) as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"{site['site_name']}: forecast request failed after "
                    f"{max_retries} attempts: {exc}"
                ) from exc
            wait_seconds = 2 ** (attempt - 1)
            print(
                f"  {site['site_name']}: attempt {attempt} failed "
                f"({type(exc).__name__}); retrying in {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)

    raise RuntimeError("unreachable retry state")


def validate_forecast_payload(payload: Any, hourly_variables: list[str]) -> None:
    """Confirm the response has an internally consistent hourly forecast table."""
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object response, received {type(payload).__name__}")
    if payload.get("error"):
        raise ValueError(payload.get("reason") or "Open-Meteo returned an error response")

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("response is missing hourly forecast data")
    timestamps = hourly.get("time")
    if not isinstance(timestamps, list) or not timestamps:
        raise ValueError("hourly forecast is missing non-empty time values")

    expected_length = len(timestamps)
    missing = [variable for variable in hourly_variables if variable not in hourly]
    if missing:
        raise ValueError(f"response is missing requested hourly variables: {missing}")

    malformed = [
        variable
        for variable in hourly_variables
        if not isinstance(hourly[variable], list) or len(hourly[variable]) != expected_length
    ]
    if malformed:
        raise ValueError(
            "hourly variable length does not match time values: " + ", ".join(malformed)
        )


def normalise_forecast(
    site: dict[str, Any],
    payload: dict[str, Any],
    retrieved_at_utc: str,
    hourly_variables: list[str],
) -> list[dict[str, Any]]:
    """Flatten one Open-Meteo hourly response to analysis-ready rows."""
    hourly = payload["hourly"]
    units = payload.get("hourly_units", {})
    rows: list[dict[str, Any]] = []

    for index, local_timestamp in enumerate(hourly["time"]):
        row: dict[str, Any] = {
            "retrieved_at_utc": retrieved_at_utc,
            "source": "Open-Meteo Forecast API",
            "site_id": site["site_id"],
            "site_name": site["site_name"],
            "region": site["region"],
            "latitude": site["latitude"],
            "longitude": site["longitude"],
            "forecast_timestamp_local": local_timestamp,
            "forecast_timezone": payload.get("timezone"),
            "forecast_timezone_abbreviation": payload.get("timezone_abbreviation"),
            "model_latitude": payload.get("latitude"),
            "model_longitude": payload.get("longitude"),
            "model_elevation_m": payload.get("elevation"),
        }
        for variable in hourly_variables:
            row[variable] = hourly[variable][index]
            row[f"{variable}_unit"] = units.get(variable)
            alias = FEATURE_ALIASES.get(variable)
            if alias:
                row[alias] = hourly[variable][index]
        rows.append(row)
    return rows


def write_json_atomic(path: Path, content: dict[str, Any]) -> None:
    """Write a JSON archive atomically to avoid partially written snapshots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, ensure_ascii=False)
    temporary_path.replace(path)


def write_parquet_outputs(
    rows: list[dict[str, Any]],
    processed_dir: Path,
    latest_path: Path | None,
    run_id: str,
) -> tuple[Path, Path | None]:
    """Write a dated Parquet snapshot and, if requested, a latest-copy table."""
    frame = pd.DataFrame(rows)
    run_date = run_id[:8]
    output_dir = processed_dir / f"run_date={run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    dated_path = output_dir / f"weather_forecasts_{run_id}.parquet"
    frame.to_parquet(dated_path, index=False)

    if latest_path is not None:
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = latest_path.with_suffix(latest_path.suffix + ".tmp")
        frame.to_parquet(temporary_path, index=False)
        temporary_path.replace(latest_path)
        return dated_path, latest_path
    return dated_path, None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sites-file",
        type=Path,
        default=Path("data/raw/sites.json"),
        help="JSON station registry (default: data/raw/sites.json).",
    )
    parser.add_argument(
        "--station",
        action="append",
        type=parse_station_argument,
        default=[],
        metavar="SITE_ID,NAME,LATITUDE,LONGITUDE[,REGION]",
        help="Add a station directly; may be specified more than once.",
    )
    parser.add_argument(
        "--include-all-sites",
        action="store_true",
        help="Do not limit stations from --sites-file to the three Greater Sydney regions.",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=7,
        choices=range(1, 17),
        metavar="1-16",
        help="Number of future forecast days to retrieve (default: 7).",
    )
    parser.add_argument(
        "--past-days",
        type=int,
        default=1,
        choices=range(0, 93),
        metavar="0-92",
        help="Number of preceding days to retain (default: 1).",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help=f"IANA timezone for forecast timestamps (default: {DEFAULT_TIMEZONE}).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/weather"),
        help="Directory for archived API responses.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/weather_forecasts"),
        help="Directory for dated normalised Parquet snapshots.",
    )
    parser.add_argument(
        "--latest-path",
        type=Path,
        default=Path("data/processed/weather_forecasts_latest.parquet"),
        help="Convenience Parquet path overwritten by each successful run; use an empty value to disable.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout per request in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum request attempts per station (default: {DEFAULT_MAX_RETRIES}).",
    )
    return parser


def main() -> None:
    """Fetch forecasts, persist raw and normalised data, and report run status."""
    args = build_parser().parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive")

    sites = list(args.station)
    skipped: list[str] = []
    if args.sites_file.exists():
        loaded_sites, skipped = load_sites(args.sites_file, args.include_all_sites)
        sites.extend(loaded_sites)
    elif not sites:
        raise FileNotFoundError(
            f"Site registry not found: {args.sites_file}. Provide --station or a valid --sites-file."
        )

    # Deduplicate manual and registry sites, retaining the first occurrence.
    unique_sites: list[dict[str, Any]] = []
    seen: set[tuple[str | None, float, float]] = set()
    for site in sites:
        identity = (site["site_id"], site["latitude"], site["longitude"])
        if identity not in seen:
            unique_sites.append(site)
            seen.add(identity)

    if not unique_sites:
        raise ValueError("No valid stations available after filtering.")

    run_time = utc_now()
    run_id = run_time.strftime("%Y%m%dT%H%M%SZ")
    retrieved_at_utc = run_time.isoformat().replace("+00:00", "Z")
    print(
        f"Fetching {args.forecast_days}-day hourly forecasts for {len(unique_sites)} station(s) "
        f"from Open-Meteo..."
    )
    if skipped:
        print(f"  Skipped {len(skipped)} site record(s) during registry filtering.")

    session = requests.Session()
    successful_responses: list[dict[str, Any]] = []
    normalised_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for number, site in enumerate(unique_sites, start=1):
        try:
            payload = request_forecast(
                session=session,
                site=site,
                hourly_variables=DEFAULT_HOURLY_VARIABLES,
                forecast_days=args.forecast_days,
                past_days=args.past_days,
                timezone_name=args.timezone,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            successful_responses.append({"station": site, "response": payload})
            station_rows = normalise_forecast(
                site=site,
                payload=payload,
                retrieved_at_utc=retrieved_at_utc,
                hourly_variables=DEFAULT_HOURLY_VARIABLES,
            )
            normalised_rows.extend(station_rows)
            print(
                f"  [{number}/{len(unique_sites)}] {site['site_name']}: "
                f"{len(station_rows)} hourly rows"
            )
        except RuntimeError as exc:
            failures.append(str(exc))
            print(f"  [{number}/{len(unique_sites)}] FAILED — {exc}", file=sys.stderr)

    archive = {
        "schema_version": 1,
        "source": "Open-Meteo Forecast API",
        "retrieved_at_utc": retrieved_at_utc,
        "request": {
            "endpoint": OPEN_METEO_FORECAST_URL,
            "hourly_variables": DEFAULT_HOURLY_VARIABLES,
            "forecast_days": args.forecast_days,
            "past_days": args.past_days,
            "timezone": args.timezone,
            "wind_speed_unit": "ms",
        },
        "station_count_requested": len(unique_sites),
        "station_count_succeeded": len(successful_responses),
        "station_count_failed": len(failures),
        "failures": failures,
        "responses": successful_responses,
    }
    raw_path = args.raw_dir / f"weather_forecasts_{run_id}.json"
    write_json_atomic(raw_path, archive)
    print(f"Raw archive: {raw_path}")

    if normalised_rows:
        latest_path = args.latest_path if str(args.latest_path) else None
        dated_path, current_path = write_parquet_outputs(
            rows=normalised_rows,
            processed_dir=args.processed_dir,
            latest_path=latest_path,
            run_id=run_id,
        )
        print(f"Normalised snapshot: {dated_path}")
        if current_path is not None:
            print(f"Latest snapshot: {current_path}")
        print(f"Total weather rows: {len(normalised_rows):,}")

    if failures:
        raise SystemExit(
            f"Completed with {len(failures)} failed station request(s); raw successful responses were archived."
        )
    if not normalised_rows:
        raise SystemExit("No forecast rows were retrieved.")


if __name__ == "__main__":
    main()
