# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:42:41 2023

@author: vincentkuo
"""

import asyncio
import websockets

async def hello(websocket, path):
    name = await websocket.recv()
    print(f"< {name}")

    greeting = f"Hi {name}!"

    await websocket.send(greeting)
    print(f"> {greeting}")

start_server = websockets.serve(hello, "localhost", 8888)

asyncio.get_event_loop().run_until_complete(start_server)
print("Server is running port:8888")
asyncio.get_event_loop().run_forever()

import asyncio
import websockets
async def hello():
    uri = "ws://localhost:8888"
    async with websockets.connect(uri) as websocket:
        name = input("Your name?")

        await websocket.send(name)
        print(f"> {name}")

        greeting = await websocket.recv()
        print(f"< {greeting}")

asyncio.get_event_loop().run_until_complete(hello())