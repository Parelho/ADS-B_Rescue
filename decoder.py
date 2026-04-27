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

            icao = pms.adsb.icao(msg)
            aircraft = {
                "icao": icao,
                "lat": None,
                "lon": None,
                "speed": None,
                "heading": None,
                "vertical_rate": None,
                "speed_type": None
            }

            tc = pms.adsb.typecode(msg)

            if 5 <= tc <= 18:
                lat, lon = pms.adsb.position_with_ref(msg, det_coords[0], det_coords[1]) # aircraft positon based on base station position
                # print(f"lat: {lat}, lon: {lon}")
                aircraft["lat"] = lat
                aircraft["lon"] = lon
                response = (
                    supabase.table("position_data")
                    .insert({"icao": aircraft["icao"], "lat": aircraft["lat"], "lon": aircraft["lon"], "timestamp": int(time.time())})
                    .execute()
                )
                print(response)
            elif tc == 19:
                velocity = pms.adsb.velocity(msg)
                # print(f"speed: {velocity[0]}, heading: {velocity[1]}, vertical rate: {velocity[2]}, speed type: {velocity[3]}")
                aircraft["speed"] = int(velocity[0])
                aircraft["heading"] = int(velocity[1])
                aircraft["vertical_rate"] = int(velocity[2])
                aircraft["speed_type"] = velocity[3]
                response = (
                    supabase.table("velocity_data")
                    .insert({"icao": aircraft["icao"], "speed": aircraft["speed"], "heading": aircraft["heading"], "vertical_rate": aircraft["vertical_rate"], "speed_type": aircraft["speed_type"], "timestamp": int(time.time())})
                    .execute()
                )
                print(response)

            # print(aircrafts)


det_coords = [-23.647143, -46.574283]

load_dotenv()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# run new client, change the host, port, and rawtype if needed
client = ADSBClient(host='10.33.133.21', port=30002, rawtype='raw')
client.run()