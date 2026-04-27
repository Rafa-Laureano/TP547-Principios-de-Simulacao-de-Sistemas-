from __future__ import annotations

import csv
import math
import random
import re
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


# O script foi pensado para ficar na mesma pasta do executavel
# ./lzw e do arquivo iot_diversificado.txt.

BASE_DIR = Path(__file__).resolve().parent
LZW_EXEC = BASE_DIR / "lzw"
DATASET_FILE = BASE_DIR / "iot_diversificado.txt"

# Pasta principal onde os resultados serao gravados.
OUTPUT_DIR = BASE_DIR / "resultados_monte_carlo_lzw"

# Pasta temporaria usada durante as simulacoes.
# A cada execucao o script cria arquivos de entrada e saida
# temporarios para o LZW, depois apaga.
TEMP_DIR = OUTPUT_DIR / "temp"


# SEED fixa: garante que, rodando de novo, os mesmos sorteios
# aleatorios sejam reproduzidos.
SEED = 42

# Numero de repeticoes para cada valor de n.
# Aqui está uma das partes da lógica de Monte Carlo: para cada quantidade de mensagens agrupadas, o experimento
# é repetido muitas vezes com grupos aleatórios diferentes.
RUNS_PER_N = 100

# # #Faixa de n a ser testada.
# n e o número de mensagens sorteadas e agrupadas antes da
# compressao. No artigo, para IoT diversificado, a faixa foi
# de 1 ate 169.
N_MIN = 1
N_MAX = 169

# Se False, cada grupo usa mensagens sem repeticao dentro da
# mesma rodada. Como o dataset tem 300 mensagens e n vai ate
# 169, é possivel.
# Se True, uma mesma mensagem pode aparecer mais de uma vez
# no mesmo grupo sorteado.
SAMPLE_WITH_REPLACEMENT = False

# Limite de payload LoRa em SF7 mencionado no artigo.
# Nao e obrigatorio para calcular a taxa de compressao media,
# mas foi mantido porque ajuda a interpretar quantas mensagens
# tendem a caber no pacote apos a compressao.
LORA_PAYLOAD_LIMIT_BYTES = 222



# Estas classes servem apenas para organizar melhor os dados
# produzidos pelo experimento.

@dataclass
class CompressionResult:
    # Resultado de UMA compressao do LZW.
    original_size: int
    compressed_size: int
    compression_rate: float


@dataclass
class SummaryRow:
    # Resumo estatistico de TODAS as repeticoes feitas para um
    # valor especifico de n.
    n_messages: int
    runs: int
    mean_rate: float
    std_rate: float
    ci95_low: float
    ci95_high: float
    mean_original_size: float
    mean_compressed_size: float
    std_compressed_size: float
    prob_fit_222_bytes: float



# FUNCOES AUXILIARES

def check_environment() -> None:
    """
    Confere se os arquivos necessarios existem.
    Se faltar dataset ou executavel, o script ja para aqui,
    antes de iniciar a simulacao.
    """
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {DATASET_FILE}")

    if not LZW_EXEC.exists():
        raise FileNotFoundError(f"Executavel LZW nao encontrado: {LZW_EXEC}")

    if not LZW_EXEC.is_file():
        raise RuntimeError(f"O caminho do LZW nao e um arquivo valido: {LZW_EXEC}")


def load_messages(dataset_path: Path) -> List[str]:
    """
    Le o dataset linha por linha.

    Cada linha nao vazia e tratada como uma mensagem do conjunto
    de dados. Isso e importante porque o sorteio aleatorio sera
    feito sobre essa lista de mensagens.
    """
    with dataset_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("O dataset esta vazio apos remover linhas em branco.")

    return lines


def choose_messages(messages: List[str], n: int, rng: random.Random) -> List[str]:
    """
    Sorteia n mensagens do dataset.

    Aqui entra um pouco de monte carlo

    O experimento nao comprime sempre o mesmo grupo fixo.
    Para cada rodada, ele sorteia um novo conjunto de mensagens.
    Essa variacao aleatoria entre as rodadas e o que transforma
    a simulacao em um experimento de Monte Carlo, em vez de uma
    simples compressao deterministica.
    """
    if SAMPLE_WITH_REPLACEMENT:
        # Sorteio com reposicao: a mesma mensagem pode aparecer
        # mais de uma vez dentro do mesmo grupo.
        return [rng.choice(messages) for _ in range(n)]

    if n > len(messages):
        raise ValueError(
            f"Nao e possivel sortear {n} mensagens sem reposicao de um dataset com {len(messages)} linhas."
        )

    # Sorteio sem reposicao: dentro daquela rodada, cada mensagem
    # aparece no maximo uma vez.
    return rng.sample(messages, n)


def write_input_file(selected_messages: List[str], input_path: Path) -> None:
    """
    Junta as mensagens sorteadas e grava em um arquivo temporario.

    Esse arquivo temporario representa o "grupo de n mensagens"
    que sera comprimido naquela repeticao.
    """
    content = "\n".join(selected_messages) + "\n"
    input_path.write_text(content, encoding="utf-8")


# Expressao regular usada para capturar da saida do LZW:
# - tamanho original
# - tamanho comprimido
# - taxa de compressao
LZW_OUTPUT_PATTERN = re.compile(
    r"Tamanho original:\s*(\d+)\s*bytes.*?"
    r"Tamanho comprimido:\s*(\d+)\s*bytes.*?"
    r"Taxa de compressao:\s*([\d\.,]+)%",
    re.DOTALL,
)


def decode_output(data: bytes) -> str:
    """
    Tenta decodificar a saida do executavel em algumas codificacoes
    comuns. Isso foi colocado porque o executavel LZW pode nao
    imprimir em UTF-8.
    """
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def run_lzw(input_path: Path, output_path: Path) -> CompressionResult:
    """
    Executa o compressor LZW externo.

    O script chama exatamente o executavel no formato:
        ./lzw entrada saida

    Depois extrai da saida os valores informados pelo proprio LZW.
    Se por algum motivo a saida nao puder ser interpretada, o script
    ainda calcula os tamanhos olhando diretamente os arquivos.
    """
    cmd = [str(LZW_EXEC), str(input_path), str(output_path)]

    # Aqui capturamos a saida bruta do programa, sem text=True,
    # para evitar problema de codificacao.
    proc = subprocess.run(cmd, capture_output=True)

    stdout = decode_output(proc.stdout)
    stderr = decode_output(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            "Falha ao executar o LZW.\n"
            f"Comando: {' '.join(cmd)}\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        )

    match = LZW_OUTPUT_PATTERN.search(stdout)
    if not match:
        # Plano B: se o padrao do texto nao for encontrado,
        # usa diretamente o tamanho real dos arquivos.
        if not input_path.exists() or not output_path.exists():
            raise RuntimeError(
                "Nao foi possivel interpretar a saida do LZW e os arquivos de entrada/saida nao existem.\n"
                f"STDOUT do LZW:\n{stdout}"
            )

        original_size = input_path.stat().st_size
        compressed_size = output_path.stat().st_size
        compression_rate = 100.0 * (1.0 - compressed_size / original_size)

        return CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_rate=compression_rate,
        )

    # Caso normal: valores extraidos do proprio log do LZW.
    original_size = int(match.group(1))
    compressed_size = int(match.group(2))
    compression_rate = float(match.group(3).replace(",", "."))

    return CompressionResult(
        original_size=original_size,
        compressed_size=compressed_size,
        compression_rate=compression_rate,
    )


def compute_ci95(values: List[float]) -> tuple[float, float, float, float]:
    """
    Calcula:
    - media amostral
    - desvio padrao amostral
    - intervalo de confianca aproximado de 95%



    Em Monte Carlo, nao interessa apenas uma execucao isolada.
    O que interessa e repetir o experimento varias vezes e resumir
    o comportamento medio da variavel observada. Aqui, a variavel
    observada e a taxa de compressao obtida em cada repeticao.
    """
    mean_val = statistics.mean(values)
    if len(values) == 1:
        return mean_val, 0.0, mean_val, mean_val

    std_val = statistics.stdev(values)
    margin = 1.96 * std_val / math.sqrt(len(values))
    return mean_val, std_val, mean_val - margin, mean_val + margin


def save_csv(rows: List[SummaryRow], csv_path: Path) -> None:
    """
    Salva em CSV o resumo final para cada n.

    Esse arquivo e importante para analise posterior e tambem para
    documentar os resultados do trabalho.
    """
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_messages",
                "runs",
                "mean_rate_percent",
                "std_rate_percent",
                "ci95_low_percent",
                "ci95_high_percent",
                "mean_original_size_bytes",
                "mean_compressed_size_bytes",
                "std_compressed_size_bytes",
                "prob_fit_222_bytes",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.n_messages,
                    row.runs,
                    f"{row.mean_rate:.6f}",
                    f"{row.std_rate:.6f}",
                    f"{row.ci95_low:.6f}",
                    f"{row.ci95_high:.6f}",
                    f"{row.mean_original_size:.6f}",
                    f"{row.mean_compressed_size:.6f}",
                    f"{row.std_compressed_size:.6f}",
                    f"{row.prob_fit_222_bytes:.6f}",
                ]
            )


def plot_results(rows: List[SummaryRow], output_png: Path) -> None:
    """
    Gera o grafico principal do trabalho:
    taxa de compressao media x numero de mensagens.

    A curva central mostra a media amostral de cada n.
    A faixa sombreada mostra o IC de 95%.
    """
    x = [row.n_messages for row in rows]
    mean_rate = [row.mean_rate for row in rows]
    ci_low = [row.ci95_low for row in rows]
    ci_high = [row.ci95_high for row in rows]

    plt.figure(figsize=(11, 6))
    plt.plot(x, mean_rate, linewidth=2, label="Taxa media de compressao")
    plt.fill_between(x, ci_low, ci_high, alpha=0.20, label="IC de 95%")
    plt.xlabel("Numero de mensagens agrupadas (n)")
    plt.ylabel("Taxa de compressao (%)")
    plt.title("Monte Carlo - LZW no dataset IoT diversificado")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def plot_compressed_size(rows: List[SummaryRow], output_png: Path) -> None:
    """
    Gera um segundo grafico opcional: tamanho comprimido medio
    x numero de mensagens.

    Ele ajuda a visualizar em que faixa o tamanho comprimido medio
    cruza o limite de 222 bytes do LoRa SF7.
    """
    x = [row.n_messages for row in rows]
    mean_size = [row.mean_compressed_size for row in rows]

    ci_low = []
    ci_high = []
    for row in rows:
        if row.runs <= 1:
            ci_low.append(row.mean_compressed_size)
            ci_high.append(row.mean_compressed_size)
        else:
            margin = 1.96 * row.std_compressed_size / math.sqrt(row.runs)
            ci_low.append(row.mean_compressed_size - margin)
            ci_high.append(row.mean_compressed_size + margin)

    plt.figure(figsize=(11, 6))
    plt.plot(x, mean_size, linewidth=2, label="Tamanho comprimido medio")
    plt.fill_between(x, ci_low, ci_high, alpha=0.20, label="IC de 95%")
    plt.axhline(
        LORA_PAYLOAD_LIMIT_BYTES,
        linestyle="--",
        linewidth=2,
        label=f"Limite LoRa SF7 = {LORA_PAYLOAD_LIMIT_BYTES} bytes",
    )
    plt.xlabel("Numero de mensagens agrupadas (n)")
    plt.ylabel("Tamanho comprimido medio (bytes)")
    plt.title("Monte Carlo - Tamanho comprimido medio vs numero de mensagens")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


# ============================================================
# FUNCAO PRINCIPAL
# ============================================================
def main() -> None:
    """
    Aqui acontece o experimento completo.

    Resumindo a logica:
    1) carrega o dataset
    2) para cada valor de n
    3) repete o experimento RUNS_PER_N vezes
    4) em cada repeticao sorteia n mensagens aleatorias
    5) comprime o grupo sorteado com o LZW
    6) guarda a taxa de compressao obtida naquela rodada
    7) ao final, calcula media, desvio e IC95% para aquele n
    8) repete para o proximo n

    """
    check_environment()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Limpa a pasta temporaria para nao misturar arquivos de execucoes antigas.
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Gerador pseudoaleatorio com semente fixa.
    rng = random.Random(SEED)

    # Carrega as 300 mensagens do dataset.
    messages = load_messages(DATASET_FILE)
    total_messages = len(messages)

    # Ajusta o maximo de n caso o sorteio seja sem reposicao.
    max_n = min(N_MAX, total_messages) if not SAMPLE_WITH_REPLACEMENT else N_MAX
    if N_MIN < 1 or N_MIN > max_n:
        raise ValueError(f"Faixa invalida: N_MIN={N_MIN}, max_n={max_n}")

    # Lista que vai armazenar o resumo final de cada n.
    summary_rows: List[SummaryRow] = []

    print("=" * 70)
    print("Simulacao Monte Carlo - Cenario 01 (LZW, IoT diversificado)")
    print(f"Dataset: {DATASET_FILE}")
    print(f"Executavel LZW: {LZW_EXEC}")
    print(f"Mensagens no dataset: {total_messages}")
    print(f"Faixa de n: {N_MIN} ate {max_n}")
    print(f"Execucoes por n: {RUNS_PER_N}")
    print(f"Seed: {SEED}")
    print("=" * 70)

    # --------------------------------------------------------
    # LOOP EXTERNO: percorre os valores de n
    # --------------------------------------------------------
    # Para cada n, estima a taxa media de compressao.
    for n in range(N_MIN, max_n + 1):
        # Guarda os resultados de todas as repeticoes daquele n.
        rate_values: List[float] = []
        original_sizes: List[int] = []
        compressed_sizes: List[int] = []

        # ----------------------------------------------------
        # LOOP INTERNO: repeticoes Monte Carlo para esse n
        # ----------------------------------------------------
        # Cada repeticao usa um grupo aleatorio diferente.
        for run_idx in range(1, RUNS_PER_N + 1):
            # 1) Sorteia aleatoriamente n mensagens do dataset.
            selected = choose_messages(messages, n, rng)

            # 2) Monta nomes dos arquivos temporarios dessa rodada.
            input_path = TEMP_DIR / f"input_n{n:03d}_run{run_idx:03d}.txt"
            output_path = TEMP_DIR / f"output_n{n:03d}_run{run_idx:03d}.bin"

            # 3) Escreve o grupo sorteado no arquivo de entrada.
            write_input_file(selected, input_path)

            # 4) Executa o LZW e coleta os resultados.
            result = run_lzw(input_path, output_path)

            # 5) Armazena os valores da repeticao.
            rate_values.append(result.compression_rate)
            original_sizes.append(result.original_size)
            compressed_sizes.append(result.compressed_size)

            # 6) Remove os arquivos temporarios para nao lotar o disco.
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


        # Depois de fazer varias repeticoes aleatorias, resumime
        # os resultados em media, desvio e intervalo de confianca.
        mean_rate, std_rate, ci95_low, ci95_high = compute_ci95(rate_values)
        mean_original_size = statistics.mean(original_sizes)
        mean_compressed_size = statistics.mean(compressed_sizes)
        std_compressed_size = statistics.stdev(compressed_sizes) if len(compressed_sizes) > 1 else 0.0
        prob_fit_222 = sum(1 for size in compressed_sizes if size <= LORA_PAYLOAD_LIMIT_BYTES) / len(compressed_sizes)

        row = SummaryRow(
            n_messages=n,
            runs=RUNS_PER_N,
            mean_rate=mean_rate,
            std_rate=std_rate,
            ci95_low=ci95_low,
            ci95_high=ci95_high,
            mean_original_size=mean_original_size,
            mean_compressed_size=mean_compressed_size,
            std_compressed_size=std_compressed_size,
            prob_fit_222_bytes=prob_fit_222,
        )
        summary_rows.append(row)

        print(
            f"n={n:3d} | taxa media={mean_rate:8.4f}% | IC95%=({ci95_low:8.4f}, {ci95_high:8.4f}) "
            f"| tamanho comprimido medio={mean_compressed_size:8.2f} bytes | P(<=222)={prob_fit_222:.3f}"
        )

    # Caminhos dos arquivos finais.
    csv_path = OUTPUT_DIR / "resumo_monte_carlo_lzw_iot.csv"
    fig_rate_path = OUTPUT_DIR / "grafico_taxa_compressao_vs_numero_mensagens.png"
    fig_size_path = OUTPUT_DIR / "grafico_tamanho_comprimido_vs_numero_mensagens.png"

    # Salva a tabela resumo e os graficos.
    save_csv(summary_rows, csv_path)
    plot_results(summary_rows, fig_rate_path)
    plot_compressed_size(summary_rows, fig_size_path)

    # Analise adicional: maior n cujo tamanho comprimido medio
    # ainda fica abaixo do limite do pacote LoRa SF7.
    feasible_rows = [row for row in summary_rows if row.mean_compressed_size <= LORA_PAYLOAD_LIMIT_BYTES]
    print("-" * 70)
    if feasible_rows:
        best_n_mean = feasible_rows[-1].n_messages
        print(f"Maior n com tamanho comprimido medio <= {LORA_PAYLOAD_LIMIT_BYTES} bytes: {best_n_mean}")
    else:
        print(f"Nenhum n teve tamanho comprimido medio <= {LORA_PAYLOAD_LIMIT_BYTES} bytes.")

    print("-" * 70)
    print(f"CSV salvo em: {csv_path}")
    print(f"Grafico da taxa de compressao salvo em: {fig_rate_path}")
    print(f"Grafico do tamanho comprimido salvo em: {fig_size_path}")
    print("Concluido com sucesso.")


if __name__ == "__main__":
    main()