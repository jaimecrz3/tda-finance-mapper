import pandas as pd

# Ruta de entrada: el CSV del dataset 
IN_PATH = "data/NASDAQ_100_Data_From_2010_To_2021.csv"

# Ruta de salida: guardaremos un archivo Parquet (más rápido y compacto que CSV)
OUT_PATH = "data/prices_adjclose.parquet"


def main():
    # 1) Cargar el CSV
    # Este dataset en concreto viene separado por tabuladores, por eso sep="\t".
    # Si no pones sep="\t", pandas intentará usar comas y te lo leerá mal (una sola columna enorme).
    df = pd.read_csv(IN_PATH, sep="\t")

    # 2) Convertir la columna Date a tipo datetime
    # Esto permite ordenar correctamente por fecha y hacer slicing por rangos (loc["2015":"2018"], etc.)
    df["Date"] = pd.to_datetime(df["Date"])

    # 3) Transformación a “panel ancho” (wide format)
    # El dataset está en “formato largo” (long format): muchas filas con (Date, Name, Adj Close).
    # Lo convertimos a:
    #   - filas: fechas
    #   - columnas: tickers (Name)
    #   - valores: Adj Close
    #
    # Ejemplo conceptual:
    # Date        AAPL    MSFT    AMZN
    # 2010-01-04  10.2    23.1    133.4
    # 2010-01-05  10.4    23.0    131.8
    #
    # sort_index() asegura que el índice (fechas) queda ordenado ascendente.
    prices = df.pivot(index="Date", columns="Name", values="Adj Close").sort_index()

    # 4) Selección de universo “fijo” con series completas (sin NaNs)
    # En datos de acciones es MUY común que haya NaNs porque:
    # - la empresa empezó a cotizar más tarde
    # - hubo cambios en el índice (entra/sale)
    # - faltan datos puntualmente
    #
    # Aquí elegimos una versión simplificada para empezar:
    # Nos quedamos solo con tickers que tienen datos para TODAS las fechas del panel.
    #
    # prices.isna().sum() te da, para cada columna (ticker), cuántos NaNs tiene.
    # Si es 0, significa “serie completa”.
    complete_cols = prices.columns[prices.isna().sum() == 0]

    # Filtramos el DataFrame para quedarnos solo con esos tickers “completos”.
    # Esto reduce problemas en el backtest (porque no hay huecos) y evita imputaciones.
    prices_complete = prices[complete_cols]

    # 5) Prints informativos (diagnóstico)
    # - Tickes totales en el dataset
    # - Tickers con serie completa
    # - Rango de fechas final del panel filtrado
    print(f"Tickers totales: {prices.shape[1]}")
    print(f"Tickers completos (sin NaNs): {prices_complete.shape[1]}")
    print(f"Rango fechas: {prices_complete.index.min().date()} -> {prices_complete.index.max().date()}")

    # 6) Guardar en Parquet
    # Ventajas de Parquet:
    # - carga mucho más rápida que CSV
    # - ocupa menos
    # - preserva tipos (fechas, floats) sin problemas
    prices_complete.to_parquet(OUT_PATH)

    print(f"Guardado: {OUT_PATH}")


# Este bloque hace que el script se ejecute SOLO si lo llamas como:
#   python src/prepare_prices.py
# Pero no se ejecuta si lo importas desde otro módulo.
if __name__ == "__main__":
    main()
