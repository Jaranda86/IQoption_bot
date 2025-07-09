import os
from iqoptionapi.stable_api import IQ_Option
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

MODO = "PRACTICE"
MONTO = 100
ACTIVOS = ["EURUSD-OTC", "USDJPY-OTC", "GBPUSD-OTC"]
TIEMPO_EXPIRACION = 1

API = IQ_Option(EMAIL, PASSWORD)
API.connect()
API.change_balance(MODO)

modelo = None
le_resultado = LabelEncoder()
ganadas = 0
perdidas = 0

def reconectar(api):
    while not api.check_connect():
        api.connect()
        time.sleep(2)

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detectar_vela_fuerte(df):
    df["cuerpo"] = abs(df["close"] - df["open"])
    return df["cuerpo"].iloc[-1] > df["cuerpo"].mean() * 1.5

def obtener_datos(api, activo):
    for _ in range(3):
        reconectar(api)
        velas = api.get_candles(activo, 300, 100, time.time())
        if velas:
            df = pd.DataFrame(velas)
            df["timestamp"] = pd.to_datetime(df["from"], unit="s")
            df["tendencia"] = np.where(df["close"] > df["open"], "call", "put")
            df["rsi"] = calcular_rsi(df["close"])
            return df
        time.sleep(1)
    return None

def entrenar_modelo(df):
    df["hora"] = df["timestamp"].dt.hour
    df["minuto"] = df["timestamp"].dt.minute
    df = df.dropna()
    X = df[["open", "close", "minuto", "hora", "rsi"]]
    y = le_resultado.fit_transform(df["tendencia"])
    modelo = LogisticRegression()
    modelo.fit(X, y)
    return modelo

def enviar_telegram(msg):
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                  data={"chat_id": CHAT_ID, "text": msg})

def operar(activo, direccion):
    global ganadas, perdidas
    reconectar(API)
    status, id = API.buy(MONTO, activo, direccion, TIEMPO_EXPIRACION)
    if status:
        enviar_telegram(f"✅ Operación: {direccion} en {activo}")
        resultado, _ = API.check_win_v3(id)
        if resultado == "win":
            ganadas += 1
        else:
            perdidas += 1
    else:
        enviar_telegram(f"❌ Fallo operación en {activo}")

def resumen_diario():
    enviar_telegram(f"📊 Resumen diario: Ganadas: {ganadas}, Perdidas: {perdidas}")

def dentro_de_horario():
    h = datetime.now().hour
    return h >= 20 or h <= 11

while True:
    if dentro_de_horario():
        for activo in ACTIVOS:
            df = obtener_datos(API, activo)
            if df is not None and detectar_vela_fuerte(df):
                modelo = entrenar_modelo(df)
                ult = df.iloc[-1]
                entrada = np.array([[ult["open"], ult["close"], ult["timestamp"].minute, ult["timestamp"].hour, ult["rsi"]]])
                pred = modelo.predict(entrada)
                op = le_resultado.inverse_transform(pred)[0]
                operar(activo, op)
            time.sleep(60)
    else:
        resumen_diario()
        time.sleep(600)
        
