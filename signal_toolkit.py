"""
signal_toolkit
==============

Ferramentas genéricas para análise de sinais experimentais no domínio do
tempo e da frequência:

- Carregamento de arquivos de aquisição (CSV/TXT) com cabeçalho configurável
- Janelamento (Hanning, Hamming, Blackman, retangular, ...)
- FFT com escala de amplitude correta e correção do ganho da janela
- Análise de amplitude, fase e frequência instantâneas (transformada de Hilbert)
- Funções de plotagem prontas para um ou vários sinais

Uso básico::

    from signal_toolkit import Sinal

    s = Sinal.de_arquivo("meu_arquivo.txt",
                         col_tempo="Time (s)",
                         col_sinal="AI 1/AI 1 (N)",
                         skiprows=11,
                         nome="Harmônico",
                         unidade="Força (N)")

    s.plotar_tempo()
    s.plotar_espectro(janela="hann")
    s.plotar_hilbert()
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, get_window


# ---------------------------------------------------------------------------
# Carregamento de dados
# ---------------------------------------------------------------------------

def carregar_arquivo(caminho, col_tempo=0, col_sinal=1, skiprows=0,
                     sep=",", decimal="."):
    """Lê um arquivo de aquisição e retorna (tempo, sinal) como arrays.

    Parameters
    ----------
    caminho : str
        Caminho do arquivo CSV/TXT.
    col_tempo, col_sinal : str ou int
        Nome (ou índice) da coluna de tempo e da coluna do sinal.
    skiprows : int
        Número de linhas de cabeçalho a ignorar antes da linha com os
        nomes das colunas (ex.: 11 para arquivos do software de aquisição
        usado no laboratório).
    sep, decimal : str
        Separador de colunas e separador decimal do arquivo.
    """
    df = pd.read_csv(caminho, skiprows=skiprows, sep=sep, decimal=decimal)
    if isinstance(col_tempo, int):
        col_tempo = df.columns[col_tempo]
    if isinstance(col_sinal, int):
        col_sinal = df.columns[col_sinal]
    t = df[col_tempo].to_numpy(dtype=float)
    x = df[col_sinal].to_numpy(dtype=float)
    return t, x


def frequencia_amostragem(t):
    """Estima a taxa de amostragem (Hz) a partir do vetor de tempo."""
    dt = np.median(np.diff(t))
    if dt <= 0:
        raise ValueError("Vetor de tempo não é crescente; "
                         "verifique as colunas informadas.")
    return 1.0 / dt


# ---------------------------------------------------------------------------
# Janelamento
# ---------------------------------------------------------------------------

def janela(nome, N):
    """Retorna uma janela de N pontos.

    `nome` aceita qualquer janela do scipy (`"hann"`, `"hamming"`,
    `"blackman"`, `"boxcar"`, ...) e também os apelidos `"hanning"`
    (= hann) e `"retangular"` (= boxcar). `None` equivale à janela
    retangular (sem janelamento).
    """
    if nome is None:
        return np.ones(N)
    apelidos = {"hanning": "hann", "retangular": "boxcar",
                "sem": "boxcar", "nenhuma": "boxcar"}
    nome = apelidos.get(str(nome).lower(), str(nome).lower())
    return get_window(nome, N, fftbins=False)


# ---------------------------------------------------------------------------
# Espectro (FFT)
# ---------------------------------------------------------------------------

def espectro(x, fs, janela_nome=None, remover_media=True,
             corrigir_janela=True):
    """Calcula o espectro de amplitude unilateral do sinal.

    A amplitude é normalizada (2/N) de modo que uma senoide de amplitude A
    apareça com altura A no espectro. Quando uma janela é aplicada, a
    amplitude é dividida pelo ganho coerente da janela (media da janela),
    tornando os espectros com e sem janela diretamente comparáveis.

    Returns
    -------
    freqs : ndarray
        Vetor de frequências (Hz).
    amp : ndarray
        Amplitude na unidade física do sinal.
    """
    x = np.asarray(x, dtype=float)
    if remover_media:
        x = x - x.mean()
    N = len(x)
    w = janela(janela_nome, N)
    X = np.fft.rfft(x * w)
    amp = 2.0 * np.abs(X) / N
    amp[0] /= 2.0                      # componente DC não é duplicada
    if N % 2 == 0:
        amp[-1] /= 2.0                 # nem a de Nyquist
    if corrigir_janela:
        amp /= w.mean()                # correção do ganho coerente
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, amp


# ---------------------------------------------------------------------------
# Transformada de Hilbert
# ---------------------------------------------------------------------------

def analise_hilbert(t, x, remover_media=True):
    """Amplitude, fase e frequência instantâneas via sinal analítico.

    Returns
    -------
    amplitude : ndarray  -- envelope do sinal
    fase : ndarray       -- fase desdobrada (rad)
    freq_inst : ndarray  -- frequência instantânea (Hz), mesmo tamanho de t
    """
    x = np.asarray(x, dtype=float)
    if remover_media:
        x = x - x.mean()
    xa = hilbert(x)
    amplitude = np.abs(xa)
    fase = np.unwrap(np.angle(xa))
    # gradient preserva o número de pontos e é menos ruidoso que diff
    freq_inst = np.gradient(fase, t) / (2.0 * np.pi)
    return amplitude, fase, freq_inst


# ---------------------------------------------------------------------------
# Classe de conveniência
# ---------------------------------------------------------------------------

@dataclass
class Sinal:
    """Um sinal amostrado, com metadados para os gráficos."""

    tempo: np.ndarray
    dados: np.ndarray
    nome: str = "Sinal"
    unidade: str = "Amplitude"
    cor: str = None
    fs: float = field(default=None)

    def __post_init__(self):
        self.tempo = np.asarray(self.tempo, dtype=float)
        self.dados = np.asarray(self.dados, dtype=float)
        if len(self.tempo) != len(self.dados):
            raise ValueError("tempo e dados devem ter o mesmo tamanho.")
        if len(self.tempo) < 2:
            raise ValueError("O sinal precisa de pelo menos 2 amostras.")
        if self.fs is None:
            self.fs = frequencia_amostragem(self.tempo)

    # -- construção ---------------------------------------------------------

    @classmethod
    def de_arquivo(cls, caminho, col_tempo=0, col_sinal=1, skiprows=0,
                   sep=",", decimal=".", t_min=None, t_max=None, **kwargs):
        """Cria um Sinal a partir de um arquivo, com recorte opcional
        do trecho [t_min, t_max]."""
        t, x = carregar_arquivo(caminho, col_tempo, col_sinal,
                                skiprows, sep, decimal)
        s = cls(t, x, **kwargs)
        if t_min is not None or t_max is not None:
            s = s.recortar(t_min, t_max)
        return s

    def recortar(self, t_min=None, t_max=None):
        """Retorna um novo Sinal restrito ao intervalo [t_min, t_max]."""
        m = np.ones(len(self.tempo), dtype=bool)
        if t_min is not None:
            m &= self.tempo >= t_min
        if t_max is not None:
            m &= self.tempo <= t_max
        if m.sum() < 2:
            raise ValueError("O recorte resultou em menos de 2 amostras.")
        return Sinal(self.tempo[m], self.dados[m], nome=self.nome,
                     unidade=self.unidade, cor=self.cor)

    # -- processamento ------------------------------------------------------

    def janelado(self, janela_nome="hann"):
        """Retorna um novo Sinal multiplicado pela janela escolhida."""
        w = janela(janela_nome, len(self.dados))
        return Sinal(self.tempo, self.dados * w,
                     nome=f"{self.nome} ({janela_nome})",
                     unidade=self.unidade, cor=self.cor, fs=self.fs)

    def espectro(self, janela_nome=None, **kwargs):
        return espectro(self.dados, self.fs, janela_nome, **kwargs)

    def hilbert(self, **kwargs):
        return analise_hilbert(self.tempo, self.dados, **kwargs)

    # -- gráficos individuais -----------------------------------------------

    def plotar_tempo(self, ax=None, t_min=None, t_max=None):
        s = self.recortar(t_min, t_max) if (t_min or t_max) else self
        ax = ax or plt.subplots(figsize=(9, 3.5))[1]
        ax.plot(s.tempo, s.dados, color=self.cor)
        ax.set_title(self.nome)
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel(self.unidade)
        ax.grid(True)
        return ax

    def plotar_espectro(self, janela_nome=None, ax=None, f_max=None,
                        escala_log=False, **kwargs):
        freqs, amp = self.espectro(janela_nome, **kwargs)
        ax = ax or plt.subplots(figsize=(9, 3.5))[1]
        if f_max is not None:
            m = freqs <= f_max
            freqs, amp = freqs[m], amp[m]
        ax.plot(freqs, amp, color=self.cor)
        rotulo = janela_nome if janela_nome else "sem janela"
        ax.set_title(f"{self.nome} — FFT ({rotulo})")
        ax.set_xlabel("Frequência (Hz)")
        ax.set_ylabel(self.unidade)
        if escala_log:
            ax.set_yscale("log")
        ax.grid(True)
        return ax

    def plotar_hilbert(self, axs=None):
        amplitude, fase, freq_inst = self.hilbert()
        if axs is None:
            _, axs = plt.subplots(1, 3, figsize=(15, 3.5))
        axs[0].plot(self.tempo, amplitude, color=self.cor)
        axs[0].set_title(f"{self.nome} — amplitude instantânea")
        axs[0].set_ylabel(self.unidade)
        axs[1].plot(self.tempo, fase, color=self.cor)
        axs[1].set_title(f"{self.nome} — fase instantânea")
        axs[1].set_ylabel("Fase (rad)")
        axs[2].plot(self.tempo, freq_inst, color=self.cor)
        axs[2].set_title(f"{self.nome} — frequência instantânea")
        axs[2].set_ylabel("Frequência (Hz)")
        for ax in axs:
            ax.set_xlabel("Tempo (s)")
            ax.grid(True)
        return axs


# ---------------------------------------------------------------------------
# Gráficos comparativos para uma lista de sinais
# ---------------------------------------------------------------------------

def _grade(n, ncols=2):
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows),
                            squeeze=False)
    axs = axs.flat
    return fig, axs


def plotar_sinais_tempo(sinais, titulo="Sinais no domínio do tempo"):
    fig, axs = _grade(len(sinais))
    for s, ax in zip(sinais, axs):
        s.plotar_tempo(ax=ax)
    for ax in list(axs)[len(sinais):]:
        ax.set_visible(False)
    fig.suptitle(titulo, fontsize=14)
    fig.tight_layout()
    return fig


def plotar_sinais_janelados(sinais, janela_nome="hann"):
    fig, axs = _grade(len(sinais))
    for s, ax in zip(sinais, axs):
        s.janelado(janela_nome).plotar_tempo(ax=ax)
    for ax in list(axs)[len(sinais):]:
        ax.set_visible(False)
    fig.suptitle(f"Sinais janelados — {janela_nome}", fontsize=14)
    fig.tight_layout()
    return fig


def plotar_sinais_espectro(sinais, janela_nome=None, f_max=None,
                           escala_log=False):
    fig, axs = _grade(len(sinais))
    for s, ax in zip(sinais, axs):
        s.plotar_espectro(janela_nome, ax=ax, f_max=f_max,
                          escala_log=escala_log)
    for ax in list(axs)[len(sinais):]:
        ax.set_visible(False)
    rotulo = janela_nome if janela_nome else "sem janelamento"
    fig.suptitle(f"Espectros de amplitude — {rotulo}", fontsize=14)
    fig.tight_layout()
    return fig


def plotar_janelas(nomes, N=1024):
    """Compara janelas no tempo (amostras normalizadas) e na frequência."""
    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(14, 4))
    n = np.arange(N)
    for nome in nomes:
        w = janela(nome, N)
        ax_t.plot(n / (N - 1), w, label=nome)
        W = np.abs(np.fft.rfft(w, 8 * N))
        W_db = 20 * np.log10(W / W.max() + 1e-12)
        bins = np.arange(len(W)) * N / (8 * N)   # eixo em bins da FFT
        ax_f.plot(bins, W_db, label=nome)
    ax_t.set_title("Janelas no tempo")
    ax_t.set_xlabel("Tempo normalizado")
    ax_t.set_ylabel("Amplitude")
    ax_f.set_title("Resposta em frequência das janelas")
    ax_f.set_xlabel("Frequência (bins da FFT)")
    ax_f.set_ylabel("Magnitude (dB)")
    ax_f.set_xlim(0, 20)
    ax_f.set_ylim(-120, 5)
    for ax in (ax_t, ax_f):
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    return fig


def plotar_sinais_hilbert(sinais):
    fig, axs = plt.subplots(len(sinais), 3,
                            figsize=(15, 3.5 * len(sinais)), squeeze=False)
    for s, linha in zip(sinais, axs):
        s.plotar_hilbert(axs=linha)
    fig.suptitle("Transformada de Hilbert", fontsize=14)
    fig.tight_layout()
    return fig
