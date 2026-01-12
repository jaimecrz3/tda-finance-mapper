from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd


_SENTINELS = {-99.99, -999.0, -999, -99.9}


def _extract_kf_table(lines: list[str], section_title: str) -> str:
    """
    Extrae una tabla (cabecera CSV + filas yyyymm,...) desde un fichero Ken French
    que viene con texto + múltiples secciones.
    """
    # Encuentra la línea exacta del título de sección
    idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == section_title:
            idx = i
            break
    if idx is None:
        raise ValueError(f"No encuentro la sección: {section_title!r}")

    # Baja hasta la cabecera CSV (suele empezar por coma)
    j = idx + 1
    while j < len(lines) and lines[j].strip() == "":
        j += 1

    header = lines[j]
    if "," not in header:
        raise ValueError(f"Cabecera inesperada en {section_title!r}: {header[:80]!r}")

    j += 1
    rows = [header]

    # Filas de datos: comienzan por yyyymm
    while j < len(lines) and re.match(r"^\s*\d{6}\s*,", lines[j]):
        rows.append(lines[j])
        j += 1

    if len(rows) <= 1:
        raise ValueError(f"Sección {section_title!r} sin filas de datos.")

    return "\n".join(rows)


def load_kf49_returns_monthly(
    csv_path: str | Path,
    weighting: Literal["vw", "ew"] = "vw",
) -> pd.DataFrame:
    """
    Devuelve retornos mensuales en formato decimal (0.01 = 1%),
    indexados a fin de mes, columnas = 49 industrias.

    weighting:
      - "vw": Value Weighted
      - "ew": Equal Weighted
    """
    csv_path = Path(csv_path)
    lines = csv_path.read_text(errors="ignore").splitlines()

    if weighting == "vw":
        title = "Average Value Weighted Returns -- Monthly"
    elif weighting == "ew":
        title = "Average Equal Weighted Returns -- Monthly"
    else:
        raise ValueError("weighting debe ser 'vw' o 'ew'")

    table_text = _extract_kf_table(lines, title)

    df = pd.read_csv(io.StringIO(table_text))

    # Primera columna es la fecha (sale como "Unnamed: 0" u otro nombre)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "yyyymm"})

    # Limpia nombres de columnas (hay espacios en el fichero)
    df.columns = [str(c).strip() for c in df.columns]

    # Missings
    df = df.replace(list(_SENTINELS), np.nan)

    # Index fin de mes
    yyyymm = df.pop("yyyymm").astype(int)
    idx = pd.to_datetime(yyyymm.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df.index = idx

    # A numérico y a decimal
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0

    return df.sort_index()


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


def load_kf49_prices(
    csv_path: str | Path,
    weighting: Literal["vw", "ew"] = "vw",
    base: float = 100.0,
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
    require_complete_panel: bool = True,
) -> pd.DataFrame:
    """
    Loader principal para tu pipeline:
      - lee retornos Ken French (mensuales),
      - (opcional) recorta fechas,
      - (opcional) fuerza panel completo (sin NaNs en el rango),
      - convierte a 'precios' (índice sintético) para reutilizar tu TDA.

    require_complete_panel=True imita tu patrón actual de dropna(axis=1, how="any")
    sobre el rango analizado, evitando sorpresas en ventanas.
    """
    rets = load_kf49_returns_monthly(csv_path, weighting=weighting)

    if start is not None:
        start = pd.Timestamp(start)
        rets = rets.loc[start:]
    if end is not None:
        end = pd.Timestamp(end)
        rets = rets.loc[:end]

    if require_complete_panel:
        rets = rets.dropna(axis=1, how="any")

    prices = returns_to_price_index(rets, base=base)
    return prices.sort_index()
