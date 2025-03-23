import matplotlib.pyplot as plt
import glob

def plot_all_progress(pattern="output*.txt"):
    """
    Cette fonction lit les fichiers de sortie générés par chaque processus et trace
    l'évolution du temps d'avancement en fonction des time steps pour chaque rang.
    
    :param pattern: Motif des fichiers de sortie (ex: "output*.txt" pour tous les fichiers outputXXX.txt).
    """
    files = sorted(glob.glob(pattern))  # Liste des fichiers triés par nom

    plt.figure(figsize=(10, 6))

    for file in files:
        rank = int(file[-7:-4])  # Extraction du numéro de rang depuis le nom du fichier
        time_steps = []
        temps_avancement = []

        with open(file, 'r') as f:
            lines = f.readlines()[2:]  # Ignorer l'en-tête
            for line in lines:
                parts = line.split()
                if len(parts) == 2:
                    time_steps.append(float(parts[0]))
                    temps_avancement.append(float(parts[1]))

        # Tracé des données de chaque processus
        plt.plot(time_steps, temps_avancement,marker = "." ,linestyle='--', label=f"Rank {rank}")

    plt.title("Évolution du temps d'avancement pour chaque processus")
    plt.xlabel("Time step")
    plt.ylabel("Temps d'avancement")
    plt.legend()
    plt.grid(True)
    plt.savefig("all_progress.png")  
    plt.show()

plot_all_progress()
