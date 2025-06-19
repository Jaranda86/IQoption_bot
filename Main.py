from keep_alive import keep_alive
keep_alive()

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

# 🔐 Variables de entorno
EMAIL = os.getenv("E-mail")
PASSWORD = os.getenv("password")
TOKEN = os.("telegram")
CHAT_ID = os.getenv("telegram")

MODO = "PRACTICE"
MONTO = 100
ACTIVOS = ["EURUSD-OTC", "USDJPY-OTC", "GBPUSD-OTC"]
TIEMPO_EXPIRACION = 1

# 📡 Conexión
API = IQ_Option(EMAIL, PASSWORD)
API.connect()
API.change_balance(MODO)

modelo = None
le_resultado = LabelEncoder()

ganadas = 0
perdidas = 0

def reconectar(api):
    while not api.check_connect():
        print("🔄 Reconectando...")
        try:
            api.connect()
            time.sleep(2)
        except:
            time.sleep(2)

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detectar_vela_fuerte(df):
    df["cuerpo"] = abs(df["close"] - df["open"])
    promedio = df["cuerpo"].mean()
    return df["cuerpo"].iloc[-1] > promedio * 1.5

def obtener_datos(api, activo):
    for _ in range(3):
        reconectar(api)
        try:
            velas = api.get_candles(activo, 300, 100, time.time())
            if velas:
                df = pd.DataFrame(velas)
                df["timestamp"] = pd.to_datetime(df["from"], unit="s")
                df["tendencia"] = np.where(df["close"] > df["open"], "call", "put")
                df["rsi"] = calcular_rsi(df["close"])
                return df
        except:
            time.sleep(1)
    return None

def entrenar_modelo(df):
    df["hora"] = df["timestamp"].dt.hour
    df["minuto"] = df["timestamp"].dt.minute
    df = df.dropna()
    X = df[["open", "close", "minuto", "hora", "rsi"]]
    y = le_resultado.fit_transform(df["tendencia"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    return modelo

def enviar_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": mensaje}
    try:
        requests.post(url, data=data)
    except:
        print("❌ Error al enviar mensaje a Telegram")

def operar(activo, direccion):
    global ganadas, perdidas
    reconectar(API)
    status, id = API.buy(MONTO, activo, direccion, TIEMPO_EXPIRACION)
    if status:
        enviar_telegram(f"✅ Operación: {direccion} en {activo}")
        print(f"✅ Operación ejecutada: {direccion} en {activo}")
        resultado, lucro = API.check_win_v3(id)
        if resultado == "win":
            ganadas += 1
        else:
            perdidas += 1
    else:
        enviar_telegram(f"❌ Falló operación en {activo}")
        print("❌ No se ejecutó operación")

def resumen_diario():
    enviar_telegram(f"📊 Resumen diario: Ganadas: {ganadas} / Perdidas: {perdidas}")

def dentro_de_horario():
    hora = datetime.now().hour
    return hora >= 20 or hora <= 11

# 🚀 Loop
while True:
    if dentro_de_horario():
        for activo in ACTIVOS:
            datos = obtener_datos(API, activo)
            if datos is not None and detectar_vela_fuerte(datos):
                modelo = entrenar_modelo(datos)
                ult = datos.iloc[-1]
                entrada = np.array([[ult["open"], ult["close"], ult["timestamp"].minute, ult["timestamp"].hour, ult["rsi"]]])
                pred = modelo.predict(entrada)
                direccion = le_resultado.inverse_transform(pred)[0]
                operar(activo, direccion)
            time.sleep(60)
    else:
        resumen_diario()
        print("⏸ Fuera de horario. Esperando...")
        time.sleep(600)
