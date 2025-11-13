from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

WAVELENGTH_MIN = 780.0
WAVELENGTH_MAX = 1080.0


def _build_wavelength_axis(n_points: int) -> np.ndarray:
    """
    Cria o eixo de comprimentos de onda para um equipamento com n_points
    entre 780nm e 1080nm (intervalo informado no desafio).
    """
    return np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, n_points)


def _interpolate_spectra(
    spectra: np.ndarray,
    x_src: np.ndarray,
    x_target: np.ndarray,
) -> np.ndarray:
    """
    Interpola cada linha de 'spectra' (amostra) medida em x_src
    para a grade comum x_target.

    spectra: shape (n_amostras, n_pontos_originais)
    retorno: shape (n_amostras, len(x_target))
    """
    return np.vstack(
        [np.interp(x_target, x_src, row) for row in spectra]
    )


def load_spectra_dataset(
    data_dir: str | Path,
    target_points: int = 100,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Carrega os arquivos Classe_1.csv ... Classe_5.csv, trata o problema
    de granularidade diferente dos espectrofotômetros e devolve um
    único DataFrame já interpolado para uma grade comum de 'target_points'
    entre 780nm e 1080nm.

    Retorna:
        - df: DataFrame com colunas wl_000 ... wl_099 + class_id
        - x_target: eixo de comprimentos de onda (numpy array)
    """
    data_dir = Path(data_dir)
    x_target = _build_wavelength_axis(target_points)

    dfs: List[pd.DataFrame] = []

    for class_id in range(1, 6):
        file_path = data_dir / f"Classe_{class_id}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        # Arquivos não possuem header, só valores numéricos
        df_raw = pd.read_csv(file_path, header=None)
        n_points = df_raw.shape[1]

        x_src = _build_wavelength_axis(n_points)
        spectra_raw = df_raw.values  # shape (n_amostras, n_points)

        spectra_interp = _interpolate_spectra(
            spectra_raw,
            x_src=x_src,
            x_target=x_target,
        )

        df_interp = pd.DataFrame(
            spectra_interp,
            columns=[f"wl_{i:03d}" for i in range(target_points)],
        )
        df_interp["class_id"] = class_id
        dfs.append(df_interp)

    df = pd.concat(dfs, ignore_index=True)
    return df, x_target
