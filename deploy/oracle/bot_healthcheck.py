import argparse
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that the bot log is still being updated."
    )
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--max-age-seconds", type=int, default=1800)
    args = parser.parse_args()

    log_file = args.log_file
    max_age = max(60, int(args.max_age_seconds))

    if not os.path.exists(log_file):
        print(f"CRITICAL: log file not found: {log_file}")
        return 2

    try:
        mtime = os.path.getmtime(log_file)
    except OSError as exc:
        print(f"CRITICAL: could not stat log file: {exc}")
        return 2

    age = time.time() - mtime
    if age > max_age:
        print(
            f"CRITICAL: log file is stale by {int(age)}s "
            f"(threshold {max_age}s): {log_file}"
        )
        return 1

    print(
        f"OK: log file updated {int(age)}s ago "
        f"(threshold {max_age}s): {log_file}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
