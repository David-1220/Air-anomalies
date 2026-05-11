import aiohttp
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def fetch_api(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        return {"error": f"Status {response.status}"}

async def get_all_data():
    ts_id = os.getenv("THINGSPEAK_CHANNEL_ID")
    al_key = os.getenv("AIRLABS_API_KEY")
    
    urls = {
        "weather": f"https://api.thingspeak.com/channels/{ts_id}/feeds.json?results=50",
        "flights": f"https://airlabs.co/api/v9/flights?api_key={al_key}" 
    }
    
    async with aiohttp.ClientSession() as session:
        # Creamos las tareas 
        names = list(urls.keys())
        tasks = [fetch_api(session, urls[name]) for name in names]
        
        # Esperamos los resultados de las solicitudes a las APIs
        results = await asyncio.gather(*tasks)
        
        # Devolvemos un diccionario con los resultados de cada API
        return dict(zip(names, results))