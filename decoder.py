import pyModeS as pms
from pyModeS.extra.tcpclient import TcpClient
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import time

# define your custom class by extending the TcpClient
#   - implement your handle_messages() methods
class ADSBClient(TcpClient):
    def __init__(self, host, port, rawtype):
        super(ADSBClient, self).__init__(host, port, rawtype)
        self.callsigns = {}      # ICAO -> callsign
        self.buffers = {}        # ICAO -> list of pending messages

    def flush_buffer(self, icao):
        """Send all buffered messages for this ICAO to DB"""
        if icao not in self.buffers:
            return

        callsign = self.callsigns.get(icao)
        if not callsign:
            return

        for entry in self.buffers[icao]:
            if entry["type"] == "position":
                response = (
                    supabase.table("position_data")
                    .insert({
                        "icao": icao,
                        "callsign": callsign,
                        "lat": entry["lat"],
                        "lon": entry["lon"],
                        "altitude": entry["altitude"],
                        "timestamp": entry["timestamp"]
                    })
                    .execute()
                )
                print(response)

            elif entry["type"] == "velocity":
                response = (
                    supabase.table("velocity_data")
                    .insert({
                        "icao": icao,
                        "callsign": callsign,
                        "speed": entry["speed"],
                        "heading": entry["heading"],
                        "vertical_rate": entry["vertical_rate"],
                        "speed_type": entry["speed_type"],
                        "timestamp": entry["timestamp"]
                    })
                    .execute()
                )
                print(response)

        # clear buffer after flushing
        self.buffers[icao] = []

    def handle_messages(self, messages):
        # print(messages)
        for msg, ts in messages:
            msg = msg.replace("*", "")
            msg = msg.replace(";", "")
            if len(msg) != 28:  # wrong data length
                # print("Data length wrong")
                continue

            df = pms.df(msg)

            if df != 17:  # not ADSB
                print("Not ADS-B")
                continue

            if pms.crc(msg) != 0:  # CRC fail
                print("CRC failed")
                continue
            
            # print(msg)
            # with open("output.txt", "a") as f:
            #     f.write(str(msg) + "\n")
            timestamp = int(time.time())
            response = (
                supabase.table("raw_msg")
                .insert({
                    "msg": msg,
                    "timestamp": timestamp
                })
                .execute()
            )
            print(response)

            icao = pms.adsb.icao(msg)
            tc = pms.adsb.typecode(msg)

            # --- CALLSIGN (Type Codes 1–4) ---
            if 1 <= tc <= 4:
                callsign = pms.adsb.callsign(msg).strip()
                self.callsigns[icao] = callsign
                print(f"Callsign updated: {icao} -> {callsign}")

                self.flush_buffer(icao)
                continue

            callsign = self.callsigns.get(icao)

            aircraft = {
                "icao": icao,
                "lat": None,
                "lon": None,
                "altitude": None,
                "speed": None,
                "heading": None,
                "vertical_rate": None,
                "speed_type": None
            }

            # --- POSITION ---
            if 5 <= tc <= 18:
                lat, lon = pms.adsb.position_with_ref(msg, det_coords[0], det_coords[1]) # aircraft positon based on base station position
                # print(f"lat: {lat}, lon: {lon}")
                aircraft["lat"] = lat
                aircraft["lon"] = lon

                if 9 <= tc <= 18:
                    aircraft["altitude"] = pms.adsb.altitude(msg)

                if callsign:
                    response = (
                        supabase.table("position_data")
                        .insert({
                            "icao": icao,
                            "callsign": callsign,
                            "lat": lat,
                            "lon": lon,
                            "altitude": aircraft["altitude"],
                            "timestamp": timestamp
                        })
                        .execute()
                    )
                    print(response)
                else:
                    # buffer it
                    self.buffers.setdefault(icao, []).append({
                        "type": "position",
                        "lat": lat,
                        "lon": lon,
                        "altitude": aircraft["altitude"],
                        "timestamp": timestamp
                    })
                    # print("buffered")

            # --- VELOCITY ---
            elif tc == 19:
                velocity = pms.adsb.velocity(msg)
                # print(f"speed: {velocity[0]}, heading: {velocity[1]}, vertical rate: {velocity[2]}, speed type: {velocity[3]}")
                aircraft["speed"] = int(velocity[0])
                aircraft["heading"] = int(velocity[1])
                aircraft["vertical_rate"] = int(velocity[2])
                aircraft["speed_type"] = velocity[3]

                if callsign:
                    response = (
                        supabase.table("velocity_data")
                        .insert({
                            "icao": icao,
                            "callsign": callsign,
                            "speed": aircraft["speed"],
                            "heading": aircraft["heading"],
                            "vertical_rate": aircraft["vertical_rate"],
                            "speed_type": aircraft["speed_type"],
                            "timestamp": timestamp
                        })
                        .execute()
                    )
                    print(response)
                else:
                    # buffer it
                    self.buffers.setdefault(icao, []).append({
                        "type": "velocity",
                        "speed": aircraft["speed"],
                        "heading": aircraft["heading"],
                        "vertical_rate": aircraft["vertical_rate"],
                        "speed_type": aircraft["speed_type"],
                        "timestamp": timestamp
                    })
                    # print("buffered")

            # print(aircrafts)


det_coords = [-23.647143, -46.574283]

load_dotenv()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# run new client, change the host, port, and rawtype if needed
client = ADSBClient(host='10.33.133.21', port=30002, rawtype='raw')
client.run()