
import pyModeS as pms
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import time


def fetch_all_rows(table_name):
    all_data = []
    page_size = 1000
    start = 0

    while True:
        response = (
            supabase.table(table_name)
            .select("*")
            .eq("used", False)
            .range(start, start + page_size - 1)
            .execute()
        )

        data = response.data
        if not data:
            break

        all_data.extend(data)
        start += page_size

    return all_data

def expire_old_messages():
    """Mark unused messages older than 24 hours as expired."""
    cutoff_timestamp = int(time.time()) - 86400

    response = (
        supabase.table("raw_msg")
        .update({
            "used": True,
            "error": False
        })
        .eq("used", False)
        .lt("timestamp", cutoff_timestamp)
        .execute()
    )

    expired_count = len(response.data or [])
    print(
        f"Expired {expired_count} old messages "
        f"(older than 24 hours)."
    )

class ADSBClient:
    def __init__(self):
        super(ADSBClient, self).__init__()
        self.callsigns = {}  # ICAO -> callsign
        self.buffers = {}    # ICAO -> list of pending messages

        # ICAO -> latest even/odd CPR position messages
        self.cpr_cache = {}

    def mark_message(self, message, error=False, reason=""):
        """Update message status in Supabase."""
        try:
            supabase.table("raw_msg").update({
                "used": True,
                "error": error
            }).eq("id", message["id"]).execute()

            status = "ERROR" if error else "PROCESSED"
            print(
                f"Message {message['id']}: {status}"
                + (f" - {reason}" if reason else "")
            )

        except Exception as e:
            print(
                f"Failed to update message {message['id']}: {e}"
            )
            raise

    def flush_buffer(self, icao):
        """Process buffered messages once a callsign is available."""
        if icao not in self.buffers:
            return

        callsign = self.callsigns.get(icao)
        if not callsign:
            return

        # Keep entries until they have been successfully processed.
        pending = self.buffers[icao]

        while pending:
            entry = pending[0]

            if entry["type"] == "position":
                supabase.table("position_data").insert({
                    "icao": icao,
                    "callsign": callsign,
                    "lat": entry["lat"],
                    "lon": entry["lon"],
                    "altitude": entry["altitude"],
                    "timestamp": entry["timestamp"]
                }).execute()

            elif entry["type"] == "velocity":
                supabase.table("velocity_data").insert({
                    "icao": icao,
                    "callsign": callsign,
                    "speed": entry["speed"],
                    "heading": entry["heading"],
                    "vertical_rate": entry["vertical_rate"],
                    "speed_type": entry["speed_type"],
                    "timestamp": entry["timestamp"]
                }).execute()

            # Mark used only after the data insert succeeds.
            self.mark_message(
                {"id": entry["msg_id"]},
                error=False,
                reason="Buffered message processed"
            )

            paired_msg_id = entry.get("paired_msg_id")
            if paired_msg_id is not None:
                self.mark_message(
                    {"id": paired_msg_id},
                    error=False,
                    reason="Paired CPR message processed"
                )

            # Remove only the successfully processed entry.
            pending.pop(0)

        del self.buffers[icao]

    def handle_messages(self, messages):
        for message in messages:
            if message.get("used", False):
                continue

            msg = message.get("msg", "")

            # --- VALIDATE MESSAGE LENGTH ---
            if not isinstance(msg, str) or len(msg) != 28:
                self.mark_message(
                    message,
                    error=True,
                    reason="Invalid message length"
                )
                continue

            # --- VALIDATE DOWNLINK FORMAT AND CRC ---
            try:
                df = pms.df(msg)

                if df != 17:
                    self.mark_message(
                        message,
                        error=True,
                        reason="Not an ADS-B message (DF != 17)"
                    )
                    continue

                if pms.crc(msg) != 0:
                    self.mark_message(
                        message,
                        error=True,
                        reason="CRC validation failed"
                    )
                    continue

            except Exception as e:
                # Parsing/validation failure, not a database failure.
                self.mark_message(
                    message,
                    error=True,
                    reason=f"Message validation exception: {e}"
                )
                continue

            # --- EXTRACT AIRCRAFT INFORMATION ---
            try:
                raw_timestamp = message.get("timestamp")
                if raw_timestamp is None:
                    self.mark_message(
                        message,
                        error=True,
                        reason="Missing message timestamp"
                    )
                    continue
                try:
                    timestamp = int(raw_timestamp)
                except (TypeError, ValueError):
                    self.mark_message(
                        message,
                        error=True,
                        reason="Invalid message timestamp"
                    )
                    continue

                icao = pms.adsb.icao(msg)
                tc = pms.adsb.typecode(msg)

                # --- CALLSIGN: TYPE CODES 1-4 ---
                if 1 <= tc <= 4:
                    callsign = pms.adsb.callsign(msg).strip()

                    if not callsign:
                        self.mark_message(
                            message,
                            error=True,
                            reason="Empty callsign"
                        )
                        continue

                    self.callsigns[icao] = callsign
                    print(f"Callsign updated: {icao} -> {callsign}")

                    # Flush buffered messages before marking the
                    # callsign message as successfully processed.
                    self.flush_buffer(icao)

                    self.mark_message(
                        message,
                        error=False,
                        reason="Callsign processed"
                    )
                    continue

                callsign = self.callsigns.get(icao)

                # --- POSITION: TYPE CODES 5-18 ---
                if 5 <= tc <= 18:
                    # CPR decoding requires a matching even/odd pair.
                    # Surface messages (TC 5-8) need a reference position.
                    is_odd = pms.adsb.oe_flag(msg)

                    aircraft_cpr = self.cpr_cache.setdefault(
                        icao,
                        {"even": None, "odd": None}
                    )

                    entry = {
                        "msg": msg,
                        "timestamp": timestamp,
                        "altitude": (
                            pms.adsb.altitude(msg)
                            if 9 <= tc <= 18
                            else None
                        ),
                        "msg_id": message["id"]
                    }

                    key = "odd" if is_odd else "even"
                    aircraft_cpr[key] = entry

                    even = aircraft_cpr["even"]
                    odd = aircraft_cpr["odd"]

                    # Wait until both CPR formats are available.
                    if even is None or odd is None:
                        continue

                    # A pair must be reasonably close in time.
                    if abs(even["timestamp"] - odd["timestamp"]) > 10:
                        # Keep the newer frame and wait for its counterpart.
                        if even["timestamp"] < odd["timestamp"]:
                            aircraft_cpr["even"] = None
                        else:
                            aircraft_cpr["odd"] = None
                        continue

                    # Decode the pair. Surface positions require a reference.
                    lat, lon = pms.adsb.position(
                        even["msg"],
                        odd["msg"],
                        even["timestamp"],
                        odd["timestamp"],
                        lat_ref=det_coords[0] if 5 <= tc <= 8 else None,
                        lon_ref=det_coords[1] if 5 <= tc <= 8 else None
                    )

                    # Use the most recent frame's altitude and timestamp.
                    latest = max(
                        (even, odd),
                        key=lambda item: item["timestamp"]
                    )

                    altitude = latest["altitude"]
                    position_timestamp = latest["timestamp"]

                    if callsign:
                        supabase.table("position_data").insert({
                            "icao": icao,
                            "callsign": callsign,
                            "lat": lat,
                            "lon": lon,
                            "altitude": altitude,
                            "timestamp": position_timestamp
                        }).execute()

                        # Mark both CPR frames processed after the insert.
                        self.mark_message(
                            {"id": even["msg_id"]},
                            error=False,
                            reason="Even CPR position processed"
                        )
                        self.mark_message(
                            {"id": odd["msg_id"]},
                            error=False,
                            reason="Odd CPR position processed"
                        )

                    else:
                        # Buffer one decoded position, not two raw CPR frames.
                        self.buffers.setdefault(icao, []).append({
                            "msg_id": even["msg_id"],
                            "paired_msg_id": odd["msg_id"],
                            "type": "position",
                            "lat": lat,
                            "lon": lon,
                            "altitude": altitude,
                            "timestamp": position_timestamp
                        })

                    # Clear the pair so it cannot be decoded repeatedly.
                    aircraft_cpr["even"] = None
                    aircraft_cpr["odd"] = None

                # --- VELOCITY: TYPE CODE 19 ---  
                elif tc == 19:
                    velocity = pms.adsb.velocity(msg)

                    speed = (
                        int(velocity[0])
                        if velocity[0] is not None else None
                    )
                    heading = (
                        int(velocity[1])
                        if velocity[1] is not None else None
                    )
                    vertical_rate = (
                        int(velocity[2])
                        if velocity[2] is not None else None
                    )

                    # The database requires a non-null heading.
                    # Reject this message if pyModeS cannot decode one.
                    if heading is None:
                        self.mark_message(
                            message,
                            error=True,
                            reason="Velocity message has no heading"
                        )
                        continue

                    if callsign:
                        supabase.table("velocity_data").insert({
                            "icao": icao,
                            "callsign": callsign,
                            "speed": speed,
                            "heading": heading,
                            "vertical_rate": vertical_rate,
                            "speed_type": velocity[3],
                            "timestamp": timestamp
                        }).execute()

                        self.mark_message(
                            message,
                            error=False,
                            reason="Velocity processed"
                        )

                    else:
                        self.buffers.setdefault(icao, []).append({
                            "msg_id": message["id"],
                            "type": "velocity",
                            "speed": speed,
                            "heading": heading,
                            "vertical_rate": vertical_rate,
                            "speed_type": velocity[3],
                            "timestamp": timestamp
                        })

                # --- UNSUPPORTED TYPE CODE ---
                else:
                    self.mark_message(
                        message,
                        error=True,
                        reason=f"Unsupported type code: {tc}"
                    )

            except Exception as e:
                # Do not mark the message as erroneous here:
                # this could be a temporary database/API failure.
                # Leave it unused so it can be retried.
                print(
                    f"Error processing message {message['id']}: {e}"
                )


det_coords = [-23.64699287757069, -46.57313694232773]

load_dotenv()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

client = ADSBClient()

last_db_chore_timestamp = 0

while True:
    # Perform database chores every 5 minutes.
    if time.time() - last_db_chore_timestamp >= 300:
        print("Performing database chores...")
        try:
            # 1. Expire old, unprocessed messages.
            # expire_old_messages()

            # 2. Remove duplicates and rebuild trajectories.
            supabase.rpc("run_database_chores").execute()

            print("Database chores completed successfully.")

        except Exception as e:
            print(f"Error performing database chores: {e}")

        finally:
            last_db_chore_timestamp = time.time()

    # Fetch and process new ADS-B messages.
    print("Fetching raw messages...")
    messages = fetch_all_rows("raw_msg")
    client.handle_messages(messages)

    time.sleep(5)
