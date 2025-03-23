import matplotlib.pyplot as plt

def plot_single_progress(file="output_calcul.txt"):
    """
    Lit un fichier de sortie (par défaut output_calcul.txt) et trace l'évolution du temps d'avancement.
    """
    time_steps = []
    temps_avancement = []

    with open(file, 'r') as f:
        lines = f.readlines()[1:]  # Ignorer la première ligne (en-tête)

        for line in lines:
            parts = line.split()
            if len(parts) == 2:
                time_steps.append(float(parts[0]))
                temps_avancement.append(float(parts[1]))

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, temps_avancement, linestyle='--', color='b', label=f"Progress from {file}")

    plt.title("Évolution du temps d'avancement (1 rank pour le calcul 1 rank pour l'affichage)")
    plt.xlabel("Time step")
    plt.ylabel("Temps d'avancement")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file}_progress.png")
    plt.show()

plot_single_progress("output_calcul.txt")
