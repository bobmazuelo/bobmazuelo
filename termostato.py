import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "Day/hour": ["L", "M", "X", "J", "V", "S", "D"],
    "0": [20]*7,
    "1": [19]*4+[20]*2+[19],
    "2": [19]*7,
    "3": [18.5]*7,
    "4": [18.5]*7,
    "5":[19]*5+[18.5]*2,
    "6":[21]*5+[19]*2,
    "7":[21.5]*5+[19]*2,
    "8": [20]*4+[17]+[21]*2,
    "9": [20]*4+[17]+[21]*2,
    "10": [20]*4+[17]+[21]*2,
    "11": [20]*4+[17]+[21]*2,
    "12": [20]*4+[17]+[21]*2,
    "13": [20]*4+[17]+[21]*2,
    "14": [21]*7,
    "15": [22]*5+[21]*2,
    "16": [22]*5+[21]*2,
    "17": [21.5]*5+[21]*2,
    "18": [21.5]*5+[21]*2,
    "19": [21]*7,
    "20":[21]*7,
    "21": [21]*7,
    "22": [21]*7,
    "23": [21]*7}
)

df = df.set_index("Day/hour")

def ajustar_dia_6_bloques(y, k=6, min_len=2):
    """
    y: array de 24 temperaturas (una fila de tu df)
    k: nº de bloques
    min_len: mínimo nº de horas por bloque (2 evita segmentos de 1h un poco absurdos)
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    # prefijos para calcular medias y errores rápido
    prefix = np.concatenate(([0.0], np.cumsum(y)))
    prefix2 = np.concatenate(([0.0], np.cumsum(y**2)))

    def coste(i, j):
        # error de aproximar y[i:j] por su media
        s = prefix[j] - prefix[i]
        s2 = prefix2[j] - prefix2[i]
        m = s / (j - i)
        return s2 - 2*m*s + (j - i) * m**2

    # dp[kk, j] = error mínimo usando kk bloques hasta la hora j
    dp = np.full((k+1, n+1), np.inf)
    prev = np.full((k+1, n+1), -1, dtype=int)
    dp[0, 0] = 0.0

    for kk in range(1, k+1):
        for j in range(kk*min_len, n+1 - (k-kk)*min_len):
            best = np.inf
            best_i = -1
            for i in range((kk-1)*min_len, j-min_len+1):
                if j - i < min_len:
                    continue
                c = dp[kk-1, i] + coste(i, j)
                if c < best:
                    best = c
                    best_i = i
            dp[kk, j] = best
            prev[kk, j] = best_i

    # backtracking: recuperar fronteras
    boundaries = [n]
    kk, j = k, n
    while kk > 0:
        i = prev[kk, j]
        boundaries.append(i)
        j = i
        kk -= 1
    boundaries = boundaries[::-1]   # horas [h0, h1, ..., h6]

    # temperatura óptima de cada bloque (la media del tramo)
    temps = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        m = (prefix[b] - prefix[a]) / (b - a)
        temps.append(m)

    return np.array(boundaries), np.array(temps)


if  __name__ == "__main__":
    bloques_horas = {}
    bloques_temps = {}

    for day, row in df.iterrows():
        h, t = ajustar_dia_6_bloques(row.values, k=6, min_len=2)
        bloques_horas[day] = h
        bloques_temps[day] = t

    df_7x6 = pd.DataFrame.from_dict(
            bloques_temps,
            orient="index",
            columns=[f"Bloque_{i+1}" for i in range(6)]
    )

    print(df_7x6.round(1))

    for d in bloques_horas.keys():
        horas = [f"{h:02d}:00" for h in bloques_horas[d]]
        print(horas)

