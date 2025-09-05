import zstandard
import os
import json
import sys
import csv
from datetime import datetime
import logging.handlers
import traceback
import re

# ----------------------
# CONFIG (edit these)
# ----------------------
# Point this at a FOLDER to process all .zst files inside (or a single .zst file if you prefer)
input_file = r"./subreddits24"   # <— folder or single .zst
output_file = r"./out"  # <— folder or single (auto-suffixed)

# Output format: "zst" | "txt" | "csv"
# zst keeps things compact; txt/csv can explode in size if filters are broad
output_format = "txt"

# Date window
from_date = datetime.strptime("2005-01-01", "%Y-%m-%d")
to_date   = datetime.strptime("2030-12-31", "%Y-%m-%d")

# Log “bad lines” (JSON decoding issues, missing fields)
write_bad_lines = True

# ----------------------
# KEYWORDS: co-occurrence filter
# A post matches if: (any CHILD) AND (any TENSION) [AND optionally (any PARTNER)]
# ----------------------
CHILD_TERMS = [
    "baby", "newborn", "infant", "toddler", "childbirth", "postpartum", "post-partum",
    "ppd", "ppa", "postnatal", "lactation", "breastfeeding", "formula", "sleep training",
    "night feeds", "night feeding", "colic", "c-section", "cesarean", "birth trauma",
    "maternity leave", "paternity leave", "diaper", "night wakings", "baby blues"
]

TENSION_TERMS = [
    "tension", "argument", "argue", "fighting", "fight", "resentment", "anger",
    "frustration", "burnout", "exhausted", "overwhelmed", "neglect", "distance",
    "drifted apart", "separate worlds", "roommate", "roommates", "sexless", "no sex",
    "intimacy", "withholding", "affectionless", "cheating", "affair",
    "division of labor", "mental load", "housework", "chores", "unappreciated",
    "unsupported", "conflict", "stonewall", "silent treatment"
]

PARTNER_TERMS = [
    "husband", "wife", "spouse", "partner", "boyfriend", "girlfriend", "co-parent", "coparent"
]

# If True, require at least one partner term too
require_partner_term = False

# ----------------------
# LOGGING
# ----------------------
log = logging.getLogger("bot")
log.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
log_str_handler = logging.StreamHandler()
log_str_handler.setFormatter(log_formatter)
log.addHandler(log_str_handler)
if not os.path.exists("logs"):
    os.makedirs("logs")
log_file_handler = logging.handlers.RotatingFileHandler(os.path.join("logs", "bot.log"), maxBytes=1024*1024*16, backupCount=5)
log_file_handler.setFormatter(log_formatter)
log.addHandler(log_file_handler)

# ----------------------
# IO HELPERS
# ----------------------
def write_line_zst(handle, obj_min):
    handle.write(json.dumps(obj_min, ensure_ascii=False).encode('utf-8'))
    handle.write(b"\n")

def write_line_json(handle, obj_min):
    handle.write(json.dumps(obj_min, ensure_ascii=False))
    handle.write("\n")

def write_line_csv(writer, obj_min):
    # only title + selftext
    writer.writerow([obj_min.get("title",""), obj_min.get("selftext","")])

def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)

def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)  # ~128MB chunks
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line.strip(), file_handle.tell()
            buffer = lines[-1]
        reader.close()

# ----------------------
# MATCHING LOGIC
# ----------------------
def any_in_text(terms, text):
    t = text.lower()
    for term in terms:
        if term in t:
            return True
    return False

def pull_submission_text(obj):
    # only submissions have 'title'; many comment objects do not
    if "title" not in obj:
        return None, None  # indicates it's likely a comment
    title = obj.get("title") or ""
    # some link posts have is_self=False; keep them, but selftext may be empty
    body  = obj.get("selftext") or ""
    return title, body

def cooccurrence_match_on(title, body):
    text = f"{title}\n{body}".lower()
    if not text.strip():
        return False
    if not (any_in_text(CHILD_TERMS, text) and any_in_text(TENSION_TERMS, text)):
        return False
    if require_partner_term and not any_in_text(PARTNER_TERMS, text):
        return False
    return True

# ----------------------
# CORE
# ----------------------
def process_file(input_file, output_file, output_format, from_date, to_date):
    output_path = f"{output_file}.{output_format}"
    log.info(f"Input: {input_file} : Output: {output_path}")
    writer = None

    if output_format == "zst":
        handle = zstandard.ZstdCompressor().stream_writer(open(output_path, 'wb'))
    elif output_format == "txt":
        handle = open(output_path, 'w', encoding='UTF-8')
    elif output_format == "csv":
        handle = open(output_path, 'w', encoding='UTF-8', newline='')
        writer = csv.writer(handle)
        writer.writerow(["title", "selftext"])  # header
    else:
        log.error(f"Unsupported output format {output_format}")
        sys.exit()

    file_size = os.stat(input_file).st_size
    created = None
    matched_lines = 0
    bad_lines = 0
    total_lines = 0

    for line, file_bytes_processed in read_lines_zst(input_file):
        total_lines += 1
        if total_lines % 100000 == 0:
            ts = created.strftime('%Y-%m-%d %H:%M:%S') if created else "n/a"
            pct = (file_bytes_processed / file_size) * 100 if file_size else 0.0
            log.info(f"{ts} : {total_lines:,} : {matched_lines:,} : {bad_lines:,} : {file_bytes_processed:,}:{pct:.0f}%")

        try:
            obj = json.loads(line)

            # skip comments immediately (we only want submissions)
            title, body = pull_submission_text(obj)
            if title is None:
                continue

            # date filter
            created = datetime.utcfromtimestamp(int(obj.get('created_utc', 0)) or 0)
            if created < from_date or created > to_date:
                continue

            # co-occurrence filter on title + body
            if not cooccurrence_match_on(title, body):
                continue

            # prepare minimal object
            obj_min = {
                "title": title,
                "selftext": body
            }

            # write minimal record
            matched_lines += 1
            if output_format == "zst":
                write_line_zst(handle, obj_min)
            elif output_format == "csv":
                write_line_csv(writer, obj_min)
            elif output_format == "txt":
                write_line_json(handle, obj_min)

        except (KeyError, json.JSONDecodeError, ValueError, TypeError) as err:
            bad_lines += 1
            if write_bad_lines:
                log.warning(f"Parse error: {err}")
                log.warning(line)

    handle.close()
    log.info(f"Complete : {total_lines:,} : {matched_lines:,} : {bad_lines:,}")

# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    log.info("[Co-occurrence mode] CHILD ∧ TENSION"
             + (" ∧ PARTNER" if require_partner_term else ""))
    log.info(f"From date {from_date.strftime('%Y-%m-%d')} to date {to_date.strftime('%Y-%m-%d')}")
    log.info(f"Output format: {output_format}")

    input_files = []
    if os.path.isdir(input_file):
        if not os.path.exists(output_file):
            os.makedirs(output_file, exist_ok=True)
        for file in os.listdir(input_file):
            if file.lower().endswith(".zst"):
                input_name = os.path.splitext(os.path.splitext(os.path.basename(file))[0])[0]
                input_files.append((os.path.join(input_file, file), os.path.join(output_file, input_name)))
    else:
        input_files.append((input_file, output_file))

    log.info(f"Processing {len(input_files)} files")
    for file_in, file_out in input_files:
        try:
            process_file(file_in, file_out, output_format, from_date, to_date)
        except Exception as err:
            log.warning(f"Error processing {file_in}: {err}")
            log.warning(traceback.format_exc())