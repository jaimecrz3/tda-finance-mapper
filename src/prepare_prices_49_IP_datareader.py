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
        path = (1.0 + r.loc[first:]).cumprod() * base
        # Resultado: Un índice de precios equivalente a invertir base en 
        # ese activo en la primera fecha disponible.
        prices.loc[first:, c] = path

    return prices

def load_kf49_prices_from_returns(rets: pd.DataFrame, base: float = 100.0, require_complete_panel: bool = True):
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