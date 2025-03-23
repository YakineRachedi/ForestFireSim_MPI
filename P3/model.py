import numpy as np

ROW = 1
COLUMN = 0

def pseudo_random(t_index: int, t_time_step: int):
    xi = t_index * (t_time_step + 1)
    r = (48271 * xi) % 2147483647
    return r / 2147483646.

def log_factor(t_value: int):
    from math import log
    return log(1. + t_value) / log(256)

#Nouvelle version de Model qui divise la réparition entre les processeur et gere les lignes restantes
class Model:
    def __init__(self, comm_calcul, rank_calcul, nbp_calcul, t_length: float, t_discretization: int, t_wind_vector, t_start_fire_position, t_max_wind: float = 60.):
        from math import sqrt
        from numpy import linalg

        if t_discretization <= 0:
            raise ValueError("Le nombre de cases par direction doit être plus grand que zéro.")

        self.comm = comm_calcul
        self.rank = rank_calcul
        self.nbp = nbp_calcul
        self.geometry = t_discretization
        self.geometry_loc = t_discretization // nbp_calcul + (1 if rank_calcul < t_discretization % nbp_calcul else 0)
        self.distance = t_length / t_discretization
        self.wind = np.array(t_wind_vector)
        self.wind_speed = linalg.norm(self.wind)
        self.max_wind = t_max_wind
        self.tempsDavancement = 0
        self.StockTempsDavancement = []
        self.StockTimeStep = []

        self.vegetation_map_loc = 255 * np.ones(shape=(self.geometry_loc + 2, t_discretization), dtype=np.uint8)
        self.fire_map_loc = np.zeros(shape=(self.geometry_loc + 2, t_discretization), dtype=np.uint8)
	
	#On gere les lignes supplémentaires
        if (t_start_fire_position[COLUMN] >= rank_calcul * (t_discretization // nbp_calcul) + (rank_calcul if rank_calcul <= t_discretization % nbp_calcul else t_discretization % nbp_calcul) and (t_start_fire_position[COLUMN] < (rank_calcul + 1) * (t_discretization // nbp_calcul) + (rank_calcul+1 if rank_calcul+1 <= t_discretization % nbp_calcul else t_discretization % nbp_calcul))):

            self.fire_map_loc[t_start_fire_position[COLUMN]- rank_calcul * (t_discretization // nbp_calcul) - (rank_calcul if rank_calcul <= t_discretization % nbp_calcul else t_discretization % nbp_calcul)+1, t_start_fire_position[ROW]] = np.uint8(255)
            self.fire_front_loc = {(t_start_fire_position[COLUMN] - rank_calcul * (t_discretization // nbp_calcul) - (rank_calcul if rank_calcul <= t_discretization % nbp_calcul else t_discretization % nbp_calcul)+1, t_start_fire_position[ROW]): np.uint8(255)}
        else:
            self.fire_front_loc = {}	
	
        ALPHA0 = 4.52790762e-01
        ALPHA1 = 9.58264437e-04
        ALPHA2 = 3.61499382e-05

        self.p1 = 0.
        if self.wind_speed < self.max_wind:
            self.p1 = ALPHA0 + ALPHA1 * self.wind_speed + ALPHA2 * (self.wind_speed * self.wind_speed)
        else:
            self.p1 = ALPHA0 + ALPHA1 * self.max_wind + ALPHA2 * (self.max_wind * self.max_wind)
        self.p2 = 0.3

        if self.wind[COLUMN] > 0:
            self.alphaEastWest = abs(self.wind[COLUMN] / t_max_wind) + 1
            self.alphaWestEast = 1. - abs(self.wind[COLUMN] / t_max_wind)
        else:
            self.alphaWestEast = abs(self.wind[COLUMN] / t_max_wind) + 1
            self.alphaEastWest = 1. - abs(self.wind[COLUMN] / t_max_wind)

        if self.wind[ROW] > 0:
            self.alphaSouthNorth = abs(self.wind[ROW] / t_max_wind) + 1
            self.alphaNorthSouth = 1. - abs(self.wind[ROW] / self.max_wind)
        else:
            self.alphaNorthSouth = abs(self.wind[ROW] / self.max_wind) + 1
            self.alphaSouthNorth = 1. - abs(self.wind[ROW] / self.max_wind)
        self.time_step = 0

#Update spécialement pour les fantomes
    def update_fantome(self):
        if self.rank != 0:
            req1 = self.comm.Irecv(self.fire_map_loc[0, :], source=self.rank - 1, tag=100)
            self.comm.Send(self.fire_map_loc[1, :], dest=self.rank - 1, tag=101)

        if self.rank != self.nbp - 1:
            req2 = self.comm.Irecv(self.fire_map_loc[-1, :], source=self.rank + 1, tag=101)
            self.comm.Send(self.fire_map_loc[-2, :], dest=self.rank + 1, tag=100)

        if self.rank != 0:
            req3 = self.comm.Irecv(self.vegetation_map_loc[0, :], source=self.rank - 1, tag=102)
            self.comm.Send(self.vegetation_map_loc[1, :], dest=self.rank - 1, tag=103)

        if self.rank != self.nbp - 1:
            req4 = self.comm.Irecv(self.vegetation_map_loc[-1, :], source=self.rank + 1, tag=103)
            self.comm.Send(self.vegetation_map_loc[-2, :], dest=self.rank + 1, tag=102)

        if self.rank != 0:
            req1.wait()
            req3.wait()
        if self.rank != self.nbp - 1:
            req2.wait()
            req4.wait()

        dico_precedent = {(0, i): 255 for i in range(self.geometry) if self.fire_map_loc[0, i] == 255}
        dico_suivant = {(self.geometry_loc + 1, i): 255 for i in range(self.geometry) if self.fire_map_loc[self.geometry_loc + 1, i] == 255}
        self.fire_front_loc = self.fire_front_loc | dico_suivant | dico_precedent

    def glob_index(self, coord):
        return coord[ROW] * self.geometry + coord[COLUMN]
#Update pour les cellules local
    def update(self) -> bool:
        import copy, time
        next_front = copy.deepcopy(self.fire_front_loc)
        t_deb = time.time()
        for lexico_coord, fire in self.fire_front_loc.items():
            power = log_factor(fire)
            if lexico_coord[ROW] < self.geometry - 1:
                tirage = pseudo_random(self.glob_index(lexico_coord) * 4059131 + self.time_step, self.time_step)
                green_power = self.vegetation_map_loc[lexico_coord[COLUMN], lexico_coord[ROW] + 1]
                correction = power * log_factor(green_power)
                if tirage < self.alphaSouthNorth * self.p1 * correction:
                    self.fire_map_loc[lexico_coord[COLUMN], lexico_coord[ROW] + 1] = np.uint8(255)
                    next_front[(lexico_coord[COLUMN], lexico_coord[ROW] + 1)] = np.uint8(255)

            if lexico_coord[ROW] > 0:
                tirage = pseudo_random(self.glob_index(lexico_coord) * 13427 + self.time_step, self.time_step)
                green_power = self.vegetation_map_loc[lexico_coord[COLUMN], lexico_coord[ROW] - 1]
                correction = power * log_factor(green_power)
                if tirage < self.alphaNorthSouth * self.p1 * correction:
                    self.fire_map_loc[lexico_coord[COLUMN], lexico_coord[ROW] - 1] = np.uint8(255)
                    next_front[(lexico_coord[COLUMN], lexico_coord[ROW] - 1)] = np.uint8(255)

            if lexico_coord[COLUMN] < self.geometry_loc + 1:
                tirage = pseudo_random(self.glob_index(lexico_coord) + self.time_step * 42569, self.time_step)
                green_power = self.vegetation_map_loc[lexico_coord[COLUMN] + 1, lexico_coord[ROW]]
                correction = power * log_factor(green_power)
                if tirage < self.alphaEastWest * self.p1 * correction:
                    self.fire_map_loc[lexico_coord[COLUMN] + 1, lexico_coord[ROW]] = np.uint8(255)
                    next_front[(lexico_coord[COLUMN] + 1, lexico_coord[ROW])] = np.uint8(255)

            if lexico_coord[COLUMN] > 0:
                tirage = pseudo_random(self.glob_index(lexico_coord) * 13427 + self.time_step * 42569, self.time_step)
                green_power = self.vegetation_map_loc[lexico_coord[COLUMN] - 1, lexico_coord[ROW]]
                correction = power * log_factor(green_power)
                if tirage < self.alphaWestEast * self.p1 * correction:
                    self.fire_map_loc[lexico_coord[COLUMN] - 1, lexico_coord[ROW]] = np.uint8(255)
                    next_front[(lexico_coord[COLUMN] - 1, lexico_coord[ROW])] = np.uint8(255)

            # Si le feu est à son maximum
            if fire == 255:
                tirage = pseudo_random(self.glob_index(lexico_coord) * 52513 + self.time_step, self.time_step)
                if tirage < self.p2:
                    self.fire_map_loc[lexico_coord[COLUMN], lexico_coord[ROW]] >>= 1
                    next_front[(lexico_coord[COLUMN], lexico_coord[ROW])] >>= 1
            else:
                # Foyer en train de s'éteindre
                self.fire_map_loc[lexico_coord[COLUMN], lexico_coord[ROW]] >>= 1
                next_front[(lexico_coord[COLUMN], lexico_coord[ROW])] >>= 1
                if next_front[(lexico_coord[COLUMN], lexico_coord[ROW])] == 0:
                    next_front.pop((lexico_coord[COLUMN], lexico_coord[ROW]))

        # Mise à jour du front de feu
        self.fire_front_loc = next_front
        t_fin = time.time()
        self.tempsDavancement += t_fin - t_deb
        self.StockTempsDavancement.append(t_fin - t_deb)
        self.StockTimeStep.append(self.time_step)

        # Diminution de la végétation aux endroits brûlés
        for lexico_coord, _ in self.fire_front_loc.items():
            if self.vegetation_map_loc[lexico_coord[COLUMN], lexico_coord[ROW]] > 0:
                self.vegetation_map_loc[lexico_coord[COLUMN], lexico_coord[ROW]] -= 1

        self.time_step += 1
        return len(self.fire_front_loc) > 0
    def get_progress(self):
        return self.tempsDavancement

    def save_and_plot_progress(self):
        buffer_filename = f"output{self.rank:03d}.txt"

        with open(buffer_filename, 'w') as out:
            out.write(f"rank {self.rank}\n")
            out.write("Time step\tTemps d'avancement\n")
            for t, temp in zip(self.StockTimeStep, self.StockTempsDavancement):
                out.write(f"{t}\t{temp}\n")

        import matplotlib.pyplot as plt
        plt.plot(self.StockTimeStep, self.StockTempsDavancement, marker='.', linestyle='--')
        plt.title(f"Évolution du temps d'avancement (rank = {self.rank + 1})")
        plt.xlabel("Time step")
        plt.ylabel("Temps d'avancement")
        plt.grid(True)
        plt.savefig(f"progress_rank{self.rank:03d}.png")  # Sauvegarde du graphe
        plt.close()  # Évite d'afficher le graphique inutilement lors de l'exécution
