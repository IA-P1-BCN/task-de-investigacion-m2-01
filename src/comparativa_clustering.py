"""Comparativa práctica de cuatro algoritmos de clustering.

Genera las tres figuras que acompañan al README y muestra por pantalla las
tablas de métricas que se citan en él.

Uso:
    python src/comparativa_clustering.py

Las imágenes se guardan en la carpeta img/ del repositorio.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: solo guardamos archivos

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Paleta: validada para daltonismo. Los tres primeros colores se distinguen
# entre sí en todas las formas habituales de visión del color.
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"   # fondo
INK = "#0b0b0b"       # texto principal
INK_SOFT = "#52514e"  # texto secundario
GRID = "#e3e2de"      # rejilla
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # grupos 1, 2 y 3
NOISE = "#9b9a94"     # ruido de DBSCAN

SEMILLA = 42  # fijarla hace que los resultados sean siempre los mismos
IMG = Path(__file__).resolve().parent.parent / "img"


def preparar_datos():
    """Dos conjuntos de juguete con formas deliberadamente distintas.

    Se estandarizan (media 0, desviación 1) porque todos estos algoritmos
    miden distancias: sin escalar, la variable de mayor rango dominaría.
    """
    manchas, _ = make_blobs(
        n_samples=300, centers=3, cluster_std=0.9, random_state=SEMILLA
    )
    lunas, etiquetas_lunas = make_moons(
        n_samples=300, noise=0.06, random_state=SEMILLA
    )
    return {
        "manchas": StandardScaler().fit_transform(manchas),
        "lunas": StandardScaler().fit_transform(lunas),
        "lunas_verdad": etiquetas_lunas,
    }


def algoritmos(k, eps):
    """Los cuatro algoritmos configurados para el mismo número de grupos.

    DBSCAN es la excepción: no recibe k, sino el radio de vecindad `eps`.
    """
    return [
        ("K-Means", KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)),
        ("Jerárquico (Ward)", AgglomerativeClustering(n_clusters=k)),
        ("DBSCAN", DBSCAN(eps=eps, min_samples=5)),
        ("GMM", GaussianMixture(n_components=k, random_state=SEMILLA)),
    ]


def _limpiar(ax):
    """Quita marcos y ejes: en estos gráficos las coordenadas no significan nada."""
    ax.set_facecolor(SURFACE)
    for borde in ax.spines.values():
        borde.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def figura_comparativa(datos):
    """Rejilla 2x4: cada algoritmo (columna) sobre cada conjunto (fila)."""
    filas = [
        ("Manchas redondas\n(3 grupos compactos)", datos["manchas"], 3, 0.35),
        ("Lunas entrelazadas\n(2 formas alargadas)", datos["lunas"], 2, 0.30),
    ]

    fig, ejes = plt.subplots(2, 4, figsize=(13, 7), facecolor=SURFACE)
    fig.subplots_adjust(top=0.84, bottom=0.12, hspace=0.22, wspace=0.08)

    for i, (nombre_datos, X, k, eps) in enumerate(filas):
        for j, (nombre_alg, modelo) in enumerate(algoritmos(k, eps)):
            ax = ejes[i, j]
            etiquetas = modelo.fit_predict(X)

            # La silueta se calcula solo sobre los puntos agrupados:
            # DBSCAN marca el ruido con -1 y no forma parte de ningún grupo.
            agrupados = etiquetas != -1
            if len(set(etiquetas[agrupados])) > 1:
                valor = silhouette_score(X[agrupados], etiquetas[agrupados])
                pie = f"silueta {valor:.2f}"
            else:
                pie = "silueta n/d"

            for n, etq in enumerate(sorted(set(etiquetas))):
                puntos = etiquetas == etq
                if etq == -1:
                    ax.scatter(X[puntos, 0], X[puntos, 1], s=26, c=NOISE,
                               marker="x", linewidths=1.1, zorder=2)
                    continue
                ax.scatter(X[puntos, 0], X[puntos, 1], s=40,
                           c=SERIES[n % len(SERIES)], edgecolors=SURFACE,
                           linewidths=0.8, zorder=3)
                # Etiqueta en el centro del grupo: identifica cada grupo sin
                # depender solo del color.
                ax.text(X[puntos, 0].mean(), X[puntos, 1].mean(), str(n + 1),
                        color=INK, fontsize=13, fontweight="bold",
                        ha="center", va="center", zorder=5,
                        path_effects=[pe.withStroke(linewidth=3.5,
                                                    foreground=SURFACE)])

            _limpiar(ax)
            if i == 0:
                ax.set_title(nombre_alg, color=INK, fontsize=12,
                             fontweight="bold", pad=10)
            ax.text(0.5, -0.06, pie, transform=ax.transAxes, ha="center",
                    va="top", color=INK_SOFT, fontsize=9)
            if j == 0:
                ax.text(-0.08, 0.5, nombre_datos, transform=ax.transAxes,
                        rotation=90, ha="center", va="center",
                        color=INK_SOFT, fontsize=10)

    fig.suptitle("Cuatro algoritmos de clustering, dos formas de datos",
                 color=INK, fontsize=16, fontweight="bold", y=0.955)
    fig.text(0.5, 0.90,
             "Mismos datos, resultados muy distintos: cada algoritmo asume una "
             "forma de grupo.",
             ha="center", color=INK_SOFT, fontsize=11)

    leyenda = [
        Line2D([], [], marker="o", ls="", markersize=8, mfc=c, mec=SURFACE,
               label=f"Grupo {n}")
        for n, c in enumerate(SERIES, 1)
    ]
    leyenda.append(Line2D([], [], marker="x", ls="", markersize=7, color=NOISE,
                          label="Ruido (solo DBSCAN)"))
    fig.legend(handles=leyenda, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=10, labelcolor=INK_SOFT)

    destino = IMG / "comparativa_algoritmos.png"
    fig.savefig(destino, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return destino


def figura_codo(datos):
    """Dos paneles: inercia y silueta frente al número de grupos.

    Son dos escalas distintas, así que van en gráficos separados y nunca
    con dos ejes verticales en el mismo: eso induce a error al leerlo.
    """
    X = datos["manchas"]
    ks = range(1, 9)
    inercias, siluetas = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(X)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(X, km.labels_) if k > 1 else np.nan)

    fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    fig.subplots_adjust(top=0.80, bottom=0.14, wspace=0.25)

    # El último valor de cada fila es hacia dónde se desplaza la etiqueta de
    # k = 3: en la silueta el punto está arriba del todo, así que la etiqueta
    # tiene que ir por debajo para no chocar con el título.
    paneles = [
        (izq, list(ks), inercias, "Método del codo", "Inercia",
         "El descenso se frena a partir de k = 3", (16, 16)),
        (der, list(ks)[1:], siluetas[1:], "Coeficiente de silueta",
         "Silueta media", "El máximo está en k = 3", (16, -34)),
    ]

    for ax, eje_x, eje_y, titulo, etiqueta_y, comentario, desvio in paneles:
        ax.plot(eje_x, eje_y, color=SERIES[0], linewidth=2, marker="o",
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                zorder=3)
        # Resaltamos k = 3, que es la respuesta que buscan ambos métodos.
        destacado = eje_y[eje_x.index(3)]
        ax.plot([3], [destacado], marker="o", markersize=12, color=SERIES[1],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        # El halo blanco evita que la etiqueta se confunda con la línea
        # cuando queda justo encima de ella.
        ax.annotate(f"k = 3\n{destacado:.2f}", xy=(3, destacado),
                    xytext=desvio, textcoords="offset points",
                    color=INK, fontsize=10, fontweight="bold", zorder=5,
                    path_effects=[pe.withStroke(linewidth=4, foreground=SURFACE)])

        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=9)
        ax.set_xlabel("Número de grupos (k)", color=INK_SOFT, fontsize=10)
        ax.set_ylabel(etiqueta_y, color=INK_SOFT, fontsize=10)
        ax.set_title(titulo, color=INK, fontsize=12, fontweight="bold", pad=22)
        ax.text(0.5, 1.02, comentario, transform=ax.transAxes, ha="center",
                color=INK_SOFT, fontsize=9.5)

    fig.suptitle("Cómo decidir el número de grupos", color=INK, fontsize=15,
                 fontweight="bold", y=0.97)

    destino = IMG / "metodo_codo.png"
    fig.savefig(destino, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return destino


def figura_dendrograma(datos):
    """Dendrograma real sobre una muestra pequeña, para que sea legible."""
    generador = np.random.default_rng(SEMILLA)
    muestra = datos["manchas"][generador.choice(len(datos["manchas"]), 30,
                                                replace=False)]
    enlaces = linkage(muestra, method="ward")

    # Altura de corte que deja exactamente 3 grupos: entre la tercera y la
    # segunda fusión empezando por el final.
    corte = (enlaces[-3, 2] + enlaces[-2, 2]) / 2

    set_link_color_palette(SERIES)
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=SURFACE)
    fig.subplots_adjust(top=0.80, bottom=0.12)

    dendrogram(enlaces, ax=ax, color_threshold=corte,
               above_threshold_color=NOISE, no_labels=True)

    ax.axhline(corte, color=INK, linestyle="--", linewidth=1.5, zorder=1)
    ax.text(0.995, corte, f"  corte → 3 grupos  ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", color=INK, fontsize=10, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=4, foreground=SURFACE)])

    ax.set_facecolor(SURFACE)
    for lado in ("top", "right", "bottom"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.set_xticks([])
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.set_ylabel("Distancia a la que se fusionan (Ward)", color=INK_SOFT,
                  fontsize=10)
    ax.set_title("Dendrograma: el árbol de fusiones", color=INK, fontsize=15,
                 fontweight="bold", pad=24)
    ax.text(0.5, 1.03,
            "Cada hoja es una observación. Cuanto más arriba se unen dos ramas, "
            "más distintas eran.",
            transform=ax.transAxes, ha="center", color=INK_SOFT, fontsize=10)

    destino = IMG / "dendrograma.png"
    fig.savefig(destino, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return destino


def tabla_codo(datos):
    """Inercia y silueta para k de 1 a 8 sobre el conjunto de manchas."""
    X = datos["manchas"]
    print("\nElección de k (conjunto de manchas)")
    print(f"{'k':>2} {'inercia':>9} {'reducción':>10} {'silueta':>8}")
    anterior = None
    for k in range(1, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(X)
        sil = f"{silhouette_score(X, km.labels_):.3f}" if k > 1 else "n/d"
        red = f"{(anterior - km.inertia_) / anterior * 100:.1f}%" if anterior else "--"
        print(f"{k:>2} {km.inertia_:9.1f} {red:>10} {sil:>8}")
        anterior = km.inertia_


def tabla_lunas(datos):
    """Acierto real frente a silueta sobre las lunas.

    El acierto (ARI) solo puede calcularse porque son datos sintéticos de los
    que conocemos la respuesta verdadera. Con datos reales no existiría.
    """
    X, verdad = datos["lunas"], datos["lunas_verdad"]
    print("\nLunas: lo que acierta cada algoritmo frente a lo que puntúa")
    print(f"{'algoritmo':<20} {'ARI (acierto)':>14} {'silueta':>9}")
    for nombre, modelo in algoritmos(k=2, eps=0.30):
        etiquetas = modelo.fit_predict(X)
        agrupados = etiquetas != -1
        sil = silhouette_score(X[agrupados], etiquetas[agrupados])
        print(f"{nombre:<20} {adjusted_rand_score(verdad, etiquetas):>14.2f} {sil:>9.2f}")


def tabla_eps(datos):
    """Cuánto cambia DBSCAN al mover solo el radio de vecindad.

    Mismo algoritmo y mismos datos: lo único que varía es `eps`.
    """
    X, verdad = datos["lunas"], datos["lunas_verdad"]
    print("\nSensibilidad de DBSCAN al parámetro eps (lunas)")
    print(f"{'eps':>5} {'grupos':>7} {'ruido':>6} {'ARI':>6}")
    for eps in (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60):
        etiquetas = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
        grupos = len(set(etiquetas) - {-1})
        ruido = int((etiquetas == -1).sum())
        print(f"{eps:>5.2f} {grupos:>7} {ruido:>6} "
              f"{adjusted_rand_score(verdad, etiquetas):>6.2f}")


def main():
    IMG.mkdir(exist_ok=True)
    datos = preparar_datos()

    print("Generando figuras...")
    for ruta in (figura_comparativa(datos), figura_codo(datos),
                 figura_dendrograma(datos)):
        print(f"  {ruta.relative_to(IMG.parent)}")

    tabla_codo(datos)
    tabla_lunas(datos)
    tabla_eps(datos)
    print("\nListo.")


if __name__ == "__main__":
    main()
