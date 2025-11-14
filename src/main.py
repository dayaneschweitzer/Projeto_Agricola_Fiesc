from pathlib import Path

from src.data.load_data import load_spectra_dataset
from src.utils.preprocessing import split_features_target, minmax_normalize
from src.models.baseline import train_baseline_model


DATA_DIR = Path("data")
TARGET_POINTS = 100  # grade comum 780–1080 nm reamostrada em 100 pontos


def main():
    print("=== PoC Avaliação de Qualidade de Maçãs por Espectroscopia ===")

    # 1) Carregar dados das 5 classes e tratar granularidade
    print("[1/4] Carregando e interpolando espectros...")
    df, x_target = load_spectra_dataset(DATA_DIR, target_points=TARGET_POINTS)
    print(f"Total de amostras: {len(df)}")
    print(f"Número de features espectrais: {TARGET_POINTS}")
    print(f"Classes disponíveis: {sorted(df['class_id'].unique().tolist())}")

    # 2) Separar X e y
    print("[2/4] Separando features e alvo...")
    X, y = split_features_target(df, target_col="class_id")

    # 3) Normalizar
    print("[3/4] Normalizando espectros (min-max por coluna)...")
    X_norm = minmax_normalize(X)

    # 4) Treinar modelo baseline
    print("[4/4] Treinando modelo baseline (RandomForest)...")
    results = train_baseline_model(X_norm, y)

    print("\n=== Resultados da PoC (Baseline) ===")
    print(f"Acurácia no conjunto de teste: {results['accuracy']:.4f}")
    print("\nRelatório de classificação:\n")
    print(results["report"])

    # x_target pode ser usado em notebooks/gráficos explicando o eixo 780–1080 nm

if __name__ == "__main__":
    main()