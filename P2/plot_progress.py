import matplotlib.pyplot as plt

def plot_two_progress():
    """
    Lit les fichiers output1.txt et output2.txt et trace l'évolution du temps d'avancement.
    """
    files = ["output1.txt", "output2.txt"]
    colors = ["b", "r"]  # Bleu et Rouge pour différencier les courbes

    plt.figure(figsize=(10, 6))

    for file, color in zip(files, colors):
        time_steps = []
        temps_avancement = []

        with open(file, 'r') as f:
            lines = f.readlines()[1:]  # Ignorer la première ligne (en-tête)

            for line in lines:
                parts = line.split()
                if len(parts) == 2:
                    time_steps.append(float(parts[0]))
                    temps_avancement.append(float(parts[1]))

        plt.plot(time_steps, temps_avancement, linestyle='--', color=color, label=file)

    plt.title("Évolution du temps d'avancement")
    plt.xlabel("Time step")
    plt.ylabel("Temps d'avancement")
    plt.legend()
    plt.grid(True)
    plt.savefig("progress_comparison.png")
    plt.show()

plot_two_progress()
