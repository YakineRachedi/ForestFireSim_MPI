# Programme principal
import sys
import pygame as pg
import display 
import model
import time
from mpi4py import MPI

from model import Model, log_factor, pseudo_random
from display import DisplayFire

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def analyze_arg( args, dico =  {} ) : 
    if len(args)==0: return dico 

    key : str = args[0]
    if key == "-l":
        if len(args) < 2:
            raise SyntaxError("Une valeur est attendue pour la longueur du terrain !")
        dico["longueur"] = float(args[1])
        if len(args) > 2 : analyze_arg(args[2:], dico)
        return
    pos = key.find("--longueur=")
    if pos >= 0:
        subkey = key[pos + len("--longueur="):]
        dico["longueur"] = float(subkey)
        if len(args) > 1 : analyze_arg(args[1:], dico)
        return

    if key == "-n":
        if len(args) < 2:
            raise SyntaxError("Une valeur est attendue pour le nombre de cellules par direction pour la discrétisation du terrain")
        dico["discretisation"] = int(args[1])
        if len(args) > 2 : analyze_arg(args[2:], dico)
        return

    pos = key.find("--number_of_cases=")
    if pos>= 0:
        subkey = key[pos+len("--number_of_cases") : ]
        dico["discretisation"] = int(subkey)
        if len(args) > 1 : analyze_arg(args[1:], dico)
        return

    if key == "-w":
        if len(args) < 2:
            raise SyntaxError("Une paire de valeur X,Y attendue pour le vecteur du vent !")
        values = args[1]
        pos_virgule = values.find(",")
        if pos_virgule <= 0:
            raise SyntaxError("Les deux réels doivent être séparés par une virgule sans espace !")
        wx = float(args[1][:pos_virgule])
        wy = float(args[1][pos_virgule+1:])
        dico["vent"] = (wx, wy)
        if len(args) > 2 : analyze_arg(args[2:], dico)
        return

    pos = key.find("--wind=")
    if pos >= 0:
        subkey = key[pos+len("--wind=") : ]
        pos_virgule = subkey.find(",")
        if pos_virgule <= 0:
            raise SyntaxError("Les deux réels doivent être séparés par une virgule sans espace !")
        wx = float(subkey[:pos_virgule])
        wy = float(subkey[pos_virgule+1:])
        dico["vent"] =  (wx, wy)
        if len(args) > 1 : analyze_arg(args[1:], dico)
        return

    if key == "-s":
        if len(args) < 2:
            raise SyntaxError("Une paire d'indice (ligne, colonne) est attendue pour la position du foyer initial !")
        values = args[1]
        pos_virgule = values.find(",")
        if pos_virgule <= 0:
            raise SyntaxError("Les deux indices doivent être séparés par une virgule sans espace !")
        fi = int(args[1][:pos_virgule])
        fj = int(args[1][pos_virgule+1:])
        dico["debut_feu"] = (fi, fj)
        if len(args) > 2: analyze_arg(args[2:], dico)
        return

    pos = key.find("--start=");
    if pos >= 0:
        subkey = key[pos + len("--start=") : ]
        pos_virgule = subkey.find(",")
        if pos_virgule <= 0:
            raise SyntaxError("Les deux indices doivent être séparés par une virgule sans espace !")
        fi = int(subkey[:pos_virgule])
        fj = int(subkey[pos_virgule+1:])
        dico["debut_feu"] = (fi, fj)
        if len(args) > 1: analyze_arg(args[1:], dico)
        return

def parse_arguments( args, dico = {} ) -> map :
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
    analyze_arg( args, dico )
    return dico

def check_params( params : map ) -> bool :
    flag : bool = True
    if params["longueur"] <= 0:
        flag = False
        raise ValueError("[ERREUR FATALE] La longueur du terrain doit être positive et non nulle !")
    if params["discretisation"] <=0:
        flag = False
        raise ValueError("[ERREUR FATALE] Le nombre de cellules par direction doit être positive et non nulle !")
    if (params["debut_feu"][0] >= params["discretisation"]) or (params["debut_feu"][1] >= params["discretisation"]) or (params["debut_feu"][0] < 0) or (params["debut_feu"][1] < 0 ):
       flag = False
       raise ValueError("[ERREUR FATALE] Mauvais indices pour la position initiale du foyer")
    return flag

def display_params(params : map ):
    print("Parametres definis pour la simulation : ")
    print(f"\tTaille du terrain                : {params['longueur']}")
    print(f"\tNombre de cellules par direction : {params['discretisation']}")
    print(f"\tVecteur vitesse vent             : {params['vent']}")
    print(f"\tPosition initiale du foyer       : {params['debut_feu']}")

params = {
    "longueur"       : 1.,
    "discretisation" : 20,
    "vent"           : (1.,1.),
    "debut_feu"      : (10, 10)
}
parse_arguments(sys.argv[1:], params)
display_params(params)
if not check_params(params):
    print("Erreur dans les paramètres !")
    exit(0)


m = model.Model( params["longueur"], params["discretisation"], params["vent"], params["debut_feu"])

avg_update_time = 0.0
avg_display_time = 0.0
count = 0

if rank == 1:
    must_continue = True
    t_counting = 0
    while (must_continue):
        t_deb = time.time()
        must_continue = m.update()
        t_fin = time.time()
        t_counting += t_fin - t_deb 
        
        comm.send(m.fire_map, dest=0, tag=1)
        comm.send(m.vegetation_map, dest=0, tag=2)
        comm.send(must_continue, dest=0, tag=3)
    TempsDeCalcMoyen = t_counting / m.time_step
    comm.send(m.time_step, dest=0, tag=4)
    comm.send(TempsDeCalcMoyen, dest=0, tag=5)
    m.get_progress()
    m.plot_and_save_progress()
elif rank == 0:
    pg.init()
    g = display.DisplayFire(params["discretisation"])
    t_display = 0
    g.init_screen()
    t_deb = time.time()
    g.update(m.fire_map, m.vegetation_map)
    t_fin = time.time()
    t_display += t_fin - t_deb
    must_continue = True
    start = time.time()
    while (must_continue):
        fire_map = comm.recv(source=1, tag=1)
        vege_map = comm.recv(source=1, tag=2)

        t_deb = time.time()

        g.update(fire_map, vege_map)

        t_fin = time.time()
        t_display += t_fin - t_deb

        must_continue = comm.recv(source=1, tag=3)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                must_continue = False
                comm.send(False, dest=1, tag=3)
    time_step = comm.recv(source=1, tag=4)
    print(f"Temps d'affichage moyen : {t_display / time_step} s")
    time_calc_moyen = comm.recv(source=1, tag=5)
    print(f"Temps de calcul moyen : {time_calc_moyen} s")

    end = time.time()
    total = end - start
    print(f"Durée totale : {total}")
    print("Fin de la simulation")
    pg.quit()
