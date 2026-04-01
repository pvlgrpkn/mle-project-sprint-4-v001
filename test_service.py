from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent
LOG_PATH = ROOT_DIR / "test_service.log"
REQUEST_TIMEOUT = 10
STARTUP_TIMEOUT = 120


def log(message: str) -> None:
    print(message)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


def reset_log() -> None:
    LOG_PATH.write_text("", encoding="utf-8")


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def choose_test_inputs() -> tuple[int, int, int]:
    recommendations = pd.read_parquet(
        ROOT_DIR / "recommendations.parquet",
        columns=["user_id", "track_id"],
    )
    known_user = int(recommendations["user_id"].iloc[0])
    unknown_user = int(recommendations["user_id"].max()) + 1
    known_tracks = set(recommendations.loc[recommendations["user_id"] == known_user, "track_id"].tolist())

    similar_sample = pd.read_parquet(
        ROOT_DIR / "similar.parquet",
        columns=["track_id", "similar_track_id"],
    ).head(5000)

    online_seed_track: int | None = None
    for track_id, group in similar_sample.groupby("track_id", sort=False):
        if any(int(candidate) not in known_tracks for candidate in group["similar_track_id"].tolist()):
            online_seed_track = int(track_id)
            break

    if online_seed_track is None:
        raise RuntimeError("Could not find a track for the online-history test scenario.")

    return known_user, unknown_user, online_seed_track


def start_service(port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.setdefault("RECSYS_DATA_DIR", str(ROOT_DIR))
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "recommendations_service:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_for_service(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.time() + STARTUP_TIMEOUT
    last_error: Exception | None = None

    while time.time() < deadline:
        if process.poll() is not None:
            break
        try:
            response = requests.get(f"{base_url}/health", timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            log(f"Healthcheck OK: {response.json()}")
            return
        except Exception as exc:  # pragma: no cover - transient startup path
            last_error = exc
            time.sleep(1)

    service_output = ""
    if process.stdout is not None:
        service_output = process.stdout.read() or ""

    raise RuntimeError(
        "Service did not become ready in time."
        + (f" Last error: {last_error}" if last_error is not None else "")
        + (f"\nService output:\n{service_output}" if service_output.strip() else "")
    )


def stop_service(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    if process.stdout is None:
        return ""
    return process.stdout.read() or ""


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    reset_log()
    known_user, unknown_user, online_seed_track = choose_test_inputs()
    port = get_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process: subprocess.Popen[str] | None = None

    log("Starting recommendation service test run...")
    log(f"Known user with personal recommendations: {known_user}")
    log(f"Unknown user for cold-start scenario: {unknown_user}")
    log(f"Track for online-history scenario: {online_seed_track}")

    try:
        process = start_service(port)
        wait_for_service(base_url, process)

        log("Scenario 1: user without personal recommendations.")
        response = requests.get(
            f"{base_url}/recommendations",
            params={"user_id": unknown_user, "k": 5},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        sources = [item["source"] for item in payload["recommendations"]]
        ensure(len(payload["recommendations"]) == 5, "Cold-start scenario must return 5 recommendations.")
        ensure(
            all(source == "top_popular" for source in sources),
            f"Cold-start scenario must fallback to top_popular, got: {sources}",
        )
        log(f"Scenario 1 passed. Sources: {sources}")

        log("Scenario 2: user with personal recommendations and no online history.")
        response = requests.get(
            f"{base_url}/recommendations",
            params={"user_id": known_user, "k": 5},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        sources = [item["source"] for item in payload["recommendations"]]
        ensure(len(payload["recommendations"]) == 5, "Personalized scenario must return 5 recommendations.")
        ensure(
            all(source == "offline_ranked" for source in sources),
            f"Expected offline_ranked recommendations before online history, got: {sources}",
        )
        log(f"Scenario 2 passed. Sources: {sources}")

        log("Scenario 3: user with personal recommendations and online history.")
        response = requests.post(
            f"{base_url}/events",
            json={"user_id": known_user, "track_id": online_seed_track},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        log(f"Recorded event: {response.json()}")

        response = requests.get(
            f"{base_url}/recommendations",
            params={"user_id": known_user, "k": 5},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        sources = [item["source"] for item in payload["recommendations"]]
        recommended_ids = [item["track_id"] for item in payload["recommendations"]]
        ensure(len(payload["recommendations"]) == 5, "Mixed scenario must return 5 recommendations.")
        ensure(
            any(source == "online_similar" for source in sources),
            f"Expected at least one online_similar recommendation after recording history, got: {sources}",
        )
        ensure(
            online_seed_track not in recommended_ids,
            "A track from the user's online history must not appear in the recommendation list.",
        )
        log(f"Scenario 3 passed. Sources: {sources}")
        log(f"Scenario 3 recommendation IDs: {recommended_ids}")

        log("All service scenarios passed successfully.")
    except Exception as exc:
        log(f"Service test failed: {exc}")
        raise
    finally:
        if process is not None:
            service_output = stop_service(process)
            if service_output.strip():
                log("Captured service output:")
                log(service_output.strip())


if __name__ == "__main__":
    main()
