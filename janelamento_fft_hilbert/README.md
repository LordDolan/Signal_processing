# Janelamento, FFT e transformada de Hilbert

Análise de sinais experimentais generalizada para funcionar com **qualquer dado** (não apenas os arquivos do laboratório).

## O que o código faz

A partir de arquivos de aquisição (CSV/TXT com colunas de tempo e sinal), o código:

1. **Plota os sinais no domínio do tempo**, com recorte opcional do trecho de interesse;
2. **Aplica janelamento** (Hanning, Hamming, Blackman, retangular ou qualquer janela do SciPy) e mostra as janelas no tempo e na frequência;
3. **Calcula o espectro de amplitude (FFT)** dos sinais, sem e com janelamento, com escala de amplitude física correta;
4. **Faz a análise de Hilbert**: amplitude instantânea (envelope), fase instantânea e frequência instantânea de cada sinal.

## Estrutura

| Arquivo | Descrição |
|---|---|
| [`signal_toolkit.py`](signal_toolkit.py) | Módulo com todas as funções de carregamento, janelamento, FFT e Hilbert |
| [`analise_de_sinais.ipynb`](analise_de_sinais.ipynb) | Notebook de exemplo: edite a célula de configuração e rode tudo |

## Como usar com os seus dados

Instale as dependências:

```bash
pip install -r requirements.txt
```

Abra o notebook `analise_de_sinais.ipynb` e edite **apenas a célula de configuração**, descrevendo cada arquivo:

```python
ARQUIVOS = [
    dict(caminho="meu_ensaio.txt",       # caminho do arquivo CSV/TXT
         col_tempo="Time (s)",           # nome (ou índice) da coluna de tempo
         col_sinal="AI 1/AI 1 (N)",      # nome (ou índice) da coluna do sinal
         skiprows=11,                    # linhas de cabeçalho antes dos nomes das colunas
         t_min=None, t_max=0.5,          # recorte opcional do trecho analisado (s)
         nome="Meu sinal",               # rótulo dos gráficos
         unidade="Força (N)"),           # unidade do eixo y
]
JANELAS = ["hann", "hamming"]            # janelas a comparar
```

A **taxa de amostragem é estimada automaticamente** a partir da coluna de tempo. Se os arquivos configurados não existirem, o notebook roda com sinais sintéticos de demonstração (harmônico, onda quadrada, transiente e aleatório), então dá para testar tudo antes de ter dados.

Também dá para usar o módulo direto em um script:

```python
from signal_toolkit import Sinal

s = Sinal.de_arquivo("meu_ensaio.txt", col_tempo="Time (s)",
                     col_sinal="AI 1/AI 1 (N)", skiprows=11,
                     nome="Ensaio", unidade="Força (N)")

s.plotar_tempo()                 # sinal no tempo
s.plotar_espectro("hann")        # FFT com janela Hanning
s.plotar_hilbert()               # amplitude, fase e frequência instantâneas

freqs, amp = s.espectro("hann")  # ou pegue os arrays para pós-processar
```

## Métodos

- **Janelamento** — o sinal é multiplicado por uma janela (`scipy.signal.get_window`) antes da FFT para reduzir o *vazamento espectral* causado pelo truncamento do sinal. O gráfico comparativo das janelas mostra o compromisso entre largura do lóbulo principal (resolução em frequência) e altura dos lóbulos laterais (vazamento).
- **FFT (espectro de amplitude unilateral)** — via `np.fft.rfft`, com normalização `2/N` (e sem duplicar as componentes DC e de Nyquist), de modo que uma senoide de amplitude `A` apareça com altura `A` no espectro, na unidade física do sinal. A média (nível DC) é removida por padrão para não mascarar as baixas frequências.
- **Correção do ganho da janela** — a amplitude é dividida pelo *ganho coerente* da janela (média da janela), então os espectros com e sem janelamento são diretamente comparáveis.
- **Transformada de Hilbert** — o sinal analítico `x_a = x + j·H{x}` (`scipy.signal.hilbert`) fornece o envelope `|x_a|`, a fase desdobrada `unwrap(angle(x_a))` e a frequência instantânea `dφ/dt / 2π` (via `np.gradient`, que preserva o número de pontos). A frequência instantânea só tem interpretação física clara para sinais de banda estreita.

## O que sai dele

- Gráficos dos sinais no tempo (grade comparativa ou individual);
- Gráficos das janelas no tempo e da resposta em frequência das janelas (em dB);
- Gráficos dos sinais janelados;
- Espectros de amplitude sem janela e com cada janela escolhida (eixo em Hz, amplitude na unidade do sinal, opção de escala log e de limite de frequência);
- Grade com amplitude, fase e frequência instantâneas de cada sinal;
- Arrays numéricos (`freqs`, `amp`, `amplitude`, `fase`, `freq_inst`) para quem quiser pós-processar.

## Correções em relação ao notebook original

Na generalização, alguns pontos do código original foram corrigidos ou melhorados:

1. **Taxa de amostragem chumbada** — o original usava `fs = 1000` fixo para todos os sinais; agora `fs` é estimada do vetor de tempo de cada arquivo (`1/mediana(Δt)`), então sinais com taxas diferentes ficam com o eixo de frequência correto.
2. **Amplitude da FFT sem normalização** — o original plotava `|rfft(x)|` cru, cuja altura depende do número de pontos e não tem unidade física. A queda de ~44% observada ao janelar era o ganho coerente da janela (0,5 para Hanning), não um efeito físico; com a normalização `2/N` + correção do ganho, o pico lê a amplitude real da senoide com ou sem janela.
3. **Variáveis reutilizadas entre células** — `sinal`, `tempo` etc. eram redefinidos em células diferentes com recortes diferentes, então o resultado dependia da ordem de execução. Agora cada sinal é um objeto `Sinal` independente.
4. **Frequência instantânea com `np.diff`** — perdia um ponto e é mais ruidosa; trocada por `np.gradient`, que preserva o tamanho do vetor.
5. **Parâmetros de leitura chumbados** — `skiprows=11` e os nomes das colunas eram fixos; agora são parâmetros por arquivo.
6. **Nível DC** — a média do sinal agora é removida antes da FFT e da Hilbert (configurável com `remover_media=False`), evitando que um offset do sensor domine o espectro em 0 Hz e distorça o envelope.
7. **FFT das janelas em escala linear** — a comparação entre janelas ficava pouco informativa; agora a resposta em frequência das janelas é mostrada em dB, onde a diferença de lóbulos laterais entre Hanning (−31 dB) e Hamming (−43 dB) fica visível.
