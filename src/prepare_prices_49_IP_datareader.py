import pandas as pd
from pandas_datareader.data import get_data_famafrench

def returns_to_price_index(
    rets: pd.DataFrame,
    base: float = 100.0
) -> pd.DataFrame:
    """
    Convierte retornos (decimales) a un índice de precios sintético.
    Robusto a NaNs iniciales por columna (si los hubiera).
    """
    prices = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)

    for c in rets.columns:
        r = rets[c]
        # Devuelve el indice del primer valor válido no nulo en la serie
        # Es decir, busca la primera fecha donde hay un retorno válido
        first = r.first_valid_index() 
        if first is None:
            continue
        # r.loc[first:] coge los retornos desde el primer día válido
        # Si un activo tiene retornos: 0.01, -0.02, 0.03, y empezamos con base=100
        # en path se irá guardando
        # 100 * 1.01 = 101
        # 101 * 0.98 = 98.98
        # 98.98 * 1.03 = 101.9494
        #
        # El resultado muestra cómo crece una inversión inicial al reinvertir las ganancias diarias. 
        # El resultado del ejemplo anterior seria [101, 98.98, 101.9494], indicando un crecimiento total del 1.9494% al final del periodo.
        #
        # Al tenerlo en precios (multiplicar por 100) y no en retornos, nos facilita el calculo de metricas como Max Drawdown (la peor caida) 
        # por ejemplo, ya que podemos decir: En el año 2000 esto llegó a valer 350$, y en 2008 cayó a 150$. Calculo la caída sobre el precio
        path = (1.0 + r.loc[first:]).cumprod() * base # Con cumprod() multiplicamos secuencialmente cada valor con los anteriores
        # Resultado: Un índice de precios equivalente a invertir base en 
        # ese activo en la primera fecha disponible.
        prices.loc[first:, c] = path

    return prices

def load_kf49_prices_from_returns(rets: pd.DataFrame, base: float = 100.0, require_complete_panel: bool = True):
    # Busca en toda la tabla y elimina cualquier sector industrial (axis=1 significa columnas) que tenga al menos 
    # un solo mes sin datos (NaN) en toda su historia. Es necesario porque los modelos matematicos no pueden trabajar
    # con datos faltantes. Otra opcion seria imputar el dato faltante
    if require_complete_panel:
        rets = rets.dropna(axis=1, how="any")
    return returns_to_price_index(rets, base=base).sort_index()


def main():
    # -----------------------------
    # 49 Industry Portfolios
    # -----------------------------
    ds = get_data_famafrench("49_Industry_Portfolios", start="1970-01")

    ind49_monthly = (ds[0] / 100.0).round(5)
    ind49_monthly.to_csv("data/49_industries_portfolios_monthly.csv")

    # -----------------------------
    # FF3 Factors
    # -----------------------------
    ff3 = get_data_famafrench("F-F_Research_Data_Factors", start="1970-01")[0]
    ff3 = (ff3 / 100.0).round(5)
    ff3.to_csv("data/ff3_monthly.csv")

    print("Descarga completada.")

if __name__ == "__main__":
    main()