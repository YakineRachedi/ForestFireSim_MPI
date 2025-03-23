import sys
import numpy as np
import pygame as pg
import display
import model
import time
import matplotlib.pyplot as plt
from mpi4py import MPI

from model import Model, log_factor, pseudo_random
from display import DisplayFire

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
comm_calcul = comm.Split(rank == 0, rank)
rank_calcul = comm_calcul.Get_rank()
nbp_calcul = comm_calcul.Get_size()

def analyze_arg(args, dico={}):
    if len(args) == 0: return dico

    key: str = args[0]
    if key == "-l":
        if len(args) < 2:
            raise SyntaxError("Une valeur est attendue pour la longueur du terrain !")
        dico["longueur"] = float(args[1])
        if len(args) > 2: analyze_arg(args[2:], dico)
        return
    pos = key.find("--longueur=")
    if pos >= 0:
        subkey = key[pos + len("--longueur="):]
        dico["longueur"] = float(subkey)
        if len(args) > 1: analyze_arg(args[1:], dico)
        return

    if key == "-n":
        if len(args) < 2:
            raise SyntaxError("Une valeur est attendue pour le nombre de cellules par direction pour la discrétisation du terrain")
        dico["discretisation"] = int(args[1])
        if len(args) > 2: analyze_arg(args[2:], dico)
        return

    pos = key.find("--number_of_cases=")
    if pos >= 0:
        subkey = key[pos + len("--number_of_cases"):]
        dico["discretisation"] = int(subkey)
        if len(args) > 1: analyze_arg(args[1:], dico)
        return

    if key == "-w":
        if len(args) < 2:
            raise SyntaxError("Une paire de valeur X,Y attendue pour le vecteur du vent !")
        values = args[1]
        pos_virgule = values.find(",")
        if pos_virgule <= 0:
            raise SyntaxError("Les deux réels doivent être séparés par une virgule sans espace !")
        wx = float(args[1][:pos_virgule])
        wy = float(args[1][pos_virgule + 1:])
        dico["vent"] = (wx, wy)
        if len(args) > 2: analyze_arg(args[2:], dico)
        return

    pos = key.find("--wind=")
    if pos >= 0:
        subkey = key[pos + len("--wind="):]
        pos_virgule = subkey.find(",")
        if pos_virgule <= 0:
            raise SyntaxError("Les deux réels doivent être séparés par une virgule sans espace !")
        wx = float(subkey[:pos_virgule])
        wy = float(subkey[pos_virgule + 1:])
        dico["vent"] = (wx, wy)
        if len(args) > 1: analyze_arg(args[1:], dico)
        return

    if key == "-s":
        if len(args) < 2:
            raise SyntaxError("Une paire d'indice (ligne, colonne) est attendue pour la position du foyer initial !")
        values = args[1]
        pos_virgule = values.find(",")
        if pos_virgule <= 0:
            raise SyntaxError("Les deux indices doivent être séparés par une virgule sans espace !")
        fi = int(args[1][:pos_virgule])
        fj = int(args[1][pos_virgule + 1:])
        dico["debut_feu"] = (fi, fj)
        if len(args) > 2: analyze_arg(args[2:], dico)
        return

    pos = key.find("--start=")
    if pos >= 0:
        subkey = key[pos + len("--start="):]
        pos_virgule = subkey.find(",")
        if pos_virgule <= 0:
            raise SyntaxError("Les deux indices doivent être séparés par une virgule sans espace !")
        fi = int(subkey[:pos_virgule])
        fj = int(subkey[pos_virgule + 1:])
        dico["debut_feu"] = (fi, fj)
        if len(args) > 1: analyze_arg(args[1:], dico)
        return

def parse_arguments(args, dico={}) -> map:
    if len(args) == 0:
        return {}
    if args[0] == "--help" or args[0] == "-h":
        print("""
Usage : simulation [option(s)]
  Lance la simulation d'incendie en prenant en compte les [option(s)].
  Les options sont :
    -l, --longueur=LONGUEUR     Définit la taille LONGUEUR (réel en km) du carré représentant la carte de la végétation.
    -n, --number_of_cases=N     Nombre n de cases par direction pour la discrétisation
    -w, --wind=VX,VY            Définit le vecteur vitesse du vent (pas de vent par défaut).
    -s, --start=COL,ROW         Définit les indices I,J de la case où commence l'incendie ( (10,10) par défaut)

 """)
        exit(1)
    analyze_arg(args, dico)
    return dico

def check_params(params: map) -> bool:
    flag: bool = True
    if params["longueur"] <= 0:
        flag = False
        raise ValueError("[ERREUR FATALE] La longueur du terrain doit être positive et non nulle !")
    if params["discretisation"] <= 0:
        flag = False
        raise ValueError("[ERREUR FATALE] Le nombre de cellules par direction doit être positive et non nulle !")
    if (params["debut_feu"][0] >= params["discretisation"]) or (params["debut_feu"][1] >= params["discretisation"]) or (params["debut_feu"][0] < 0) or (params["debut_feu"][1] < 0):
        flag = False
        raise ValueError("[ERREUR FATALE] Mauvais indices pour la position initiale du foyer")
    return flag

def display_params(params: map):
    print("Parametres definis pour la simulation : ")
    print(f"\tTaille du terrain                : {params['longueur']}")
    print(f"\tNombre de cellules par direction : {params['discretisation']}")
    print(f"\tVecteur vitesse vent             : {params['vent']}")
    print(f"\tPosition initiale du foyer       : {params['debut_feu']}")

params = {
    "longueur": 1.,
    "discretisation": 20,
    "vent": (1., 1.),
    "debut_feu": (10, 10)
}
parse_arguments(sys.argv[1:], params)
if rank == 0: display_params(params)
if not check_params(params):
    print("Erreur dans les paramètres !")
    exit(0)

length = params["discretisation"]
# On initialise les processus
if rank != 0:
    t_calcul = 0
    m = Model(comm_calcul, rank_calcul, nbp_calcul, params["longueur"], params["discretisation"], params["vent"], params["debut_feu"])
    # Mettre à jour les cellules fantômes
    m.update_fantome()


if rank == 0:
    t_display = 0
    pg.init()
    g = DisplayFire(params["discretisation"])
    g.init_screen()

comm.Barrier()
must_continue = True
start_time = time.time()

while must_continue:
    if rank != 0:
        t_deb = time.time()
        # Mettre à jour les cellules fantômes
        m.update_fantome()
        # Mettre à jour le modèle local
        must_continue = m.update()
        t_fin = time.time()
        t_calcul += t_fin - t_deb
        # Préparer les données locales pour le Gatherv
        fire_map_loc_part = m.fire_map_loc[1:-1, :].flatten()  # Exclure les cellules fantômes
        vege_map_loc_part = m.vegetation_map_loc[1:-1, :].flatten()  # Exclure les cellules fantômes
        # Quel taille ont les tableaux
        sendcounts = np.array(comm_calcul.gather(m.fire_map_loc[1:-1,:].size, root=0))
        if rank_calcul == 0:
            # Le processeur 1 prépare la map entière
            fire_map_send = np.empty((params["discretisation"], params["discretisation"]), dtype=np.uint8)
            vege_map_send = np.empty((params["discretisation"], params["discretisation"]), dtype=np.uint8)
        else:
            # Les autres processus font rien
            fire_map_send = None
            vege_map_send = None
        # On gather toutes les map loc au processeur 1
        comm_calcul.Gatherv(fire_map_loc_part, [fire_map_send, sendcounts], root=0)
        comm_calcul.Gatherv(vege_map_loc_part, [vege_map_send, sendcounts], root=0)

        if rank_calcul == 0:
            # Send tout au proc 0
            comm.Send(fire_map_send, dest=0, tag=200)
            comm.Send(vege_map_send, dest=0, tag=201)

        # Send must_continue
        must_continue = comm_calcul.allreduce(must_continue, op=MPI.LOR)
        comm.send(must_continue, dest=0, tag=400 + rank)
        comm.send(m.time_step, dest=0, tag=500 + rank)

    # Si ce processus est le rank_calcul == 0 (c'est-à-dire rank == 1), il doit envoyer les données au rank == 0

    if rank == 0:
        # Remodeler les données en 2D
        fire_map_entier = np.empty((params["discretisation"], params["discretisation"]), dtype=np.uint8)
        vege_map_entier = np.empty((params["discretisation"], params["discretisation"]), dtype=np.uint8)

        # Recevoir les données
        comm.Recv(fire_map_entier, source=1, tag=200)
        comm.Recv(vege_map_entier, source=1, tag=201)

        t_deb = time.time()
        # Mettre à jour l'affichage
        g.update(fire_map_entier, vege_map_entier)
        t_end = time.time()
        t_display += t_end - t_deb
        # Vérifier les événements Pygame (comme la fermeture de la fenêtre)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                must_continue = False

        # Recoit l'état de must_continue si un n'a pas fini alors on continue
        must_continue = False
        for r in range(1, size):
            must_continue_t = comm.recv(source=r, tag=400 + r)
            if must_continue_t == True:
                must_continue = True
if rank != 0: 
    Temps_calcul_total = comm_calcul.allreduce(t_calcul, op=MPI.SUM)
    Nb_step = comm_calcul.allreduce(m.time_step, op=MPI.SUM)    
    temps_davancement_p = m.get_progress()
    temps_davancement_final = comm_calcul.allreduce(temps_davancement_p, op=MPI.SUM)
    if rank_calcul==0:
        comm.send(Temps_calcul_total, dest=0, tag = 1993)
        comm.send(Nb_step, dest=0, tag=2020)
        comm.send(temps_davancement_final, dest = 0, tag=3346)

    m.save_and_plot_progress()

comm_calcul.Barrier()
end = time.time()
if rank == 0:
    Temps_calcul_total = comm.recv(source=1, tag=1993)
    Nb_step = comm.recv(source=1, tag=2020)
    temps_davancement_final = comm.recv(source=1, tag=3346)
    print(f"Temps de calcul moyen : {Temps_calcul_total / Nb_step} s")
    print(f"Temps d'affichage : {t_display} s")                   
    print(f"Temps d'avancement  (par processeur) : {temps_davancement_final / (size - 1)} s")
    print(f"Temps d'avancement moyen : {temps_davancement_final / Nb_step} s")
    total = end - start_time
    print(f"Durée totale : {total}")
    total_time_step = 0
    for r in range(1, size):
        total_time_step+= comm.recv(source=r, tag=500+r)
        
    print("Fin de la simulation")
    pg.quit()
