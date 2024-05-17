import pulp
import numpy as np
import random as rd
from itertools import product
import csv
from collections import defaultdict
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import random
import matplotlib.pyplot as plt


nom_fichier = './Twitch_game_data.csv' # Nom du fichier CSV contenant les données

# Initialisation d'un dictionnaire vide
donnees = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))


with open(nom_fichier, mode='r', newline='', encoding='cp1252') as fichier_csv:
    lecteur_csv = csv.DictReader(fichier_csv)
    
    # Parcours des lignes du fichier CSV pour le remplissage du dictionnaire
    for ligne in lecteur_csv:
        game_name = ligne['Game']
        year = ligne['Year']
        month = ligne['Month']
        hours_watched = ligne['Hours_watched']
        hours_streamed = ligne['Hours_streamed']
        peak_viewers = ligne['Peak_viewers']
        peak_channels = ligne['Peak_channels']
        streamers = ligne['Streamers']
        avg_viewers = ligne['Avg_viewers']
        
        # Ajout de chaque valeur à  la liste appropriée dans le dictionnaire
        donnees[game_name][year][month] = {'hours_watched': hours_watched, 'hours_streamed': hours_streamed, 'peak_viewers': peak_viewers, 'peak_channels': peak_channels, 'streamers': streamers, 'avg_viewers': avg_viewers}

# On a un dictionnaire où chaque clé est un nom de jeu, chaque sous-clé est une année, chaque sous-sous-clé est un mois, et la valeur est un autre dictionnaire avec 'hours_watched' et 'hours_streamed'

# On décide d'enlever les jeux du dictionnaire qui moins de 4 entrées de données pour faciliter le fonctionnement du Logistic Classifier
# On crée une liste de jeux à  enlever
jeux_a_enlever = [game for game, data in donnees.items() if len(data) < 4]

# On enlève les jeux de la liste des jeux à  enlever
for game in jeux_a_enlever:
    del donnees[game]

# Maintenant le dictionnaire ne contient que des jeux avec 4 entrées de données ou plus

# On converti le dictionnaire en un dictionnaire régulier
def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d

donnees_dict = defaultdict_to_dict(donnees)

'''
# Différents tests à faire
# Affichage de tous les éléments du dictionnaire
for game_name, game_data in donnees_dict.items():
    print(f"{game_name}: {game_data}\n")

    
# Ou afficher seulement les premiers éléments de chaque jeu
for game_name, game_data in list(donnees_dict.items())[:5]:
    print(f"{game_name}: {game_data}\n")

    
# Affichage du premier élément du dictionnaire (League of Legends) afin de faire des tests avec les q retournés par remplissage_q()
first_key = next(iter(donnees_dict))
first_value = donnees_dict[first_key]


print(f"First key: {first_key}")
print(f"First value: {first_value}")
'''

def simulate_marketplace(nb_acheteurs,donnees_dict):

    

    ###############################################################################

    # Côté vendeur
    all_bids = [] # all_bids va contenir toutes les offres du marché

    q_size = 96 # Taille du vecteur q <=> nombre de types de données différents = 8*12 = 96 (une entrée pour chaque mois entre 2016 et 2023)


    # Remplissage de all_q avec les vecteurs q possibles pour chaque jeu en fonction des données présentées dans la dataset
    def remplissage_q():

        all_q = [] # all_q contient tous les vecteurs q possible (avec 0 et 1) 
        for game_name, game_data in donnees_dict.items():
            q = [game_name] + [0]*q_size # On ajoute le nom du jeu au vecteur q
            for year, year_data in game_data.items():
                year = str(year)
                k = (year-2016)*12
                for month, value in year_data.items():
                    # on remplit q en fonction des données présentes par année et mois
                    month = str(month)
                    j = month - 1               
                    
                    q[j+k+1]= 1 # +1 pour éviter de modifier la première entrée de q qui correspond au nom du jeu
            all_q.append(q)
        return all_q

    all_q = remplissage_q()

    '''
    # Différents tests à faire
    # On vérifie que tous les vecteurs q sont dans all_q 
    print(len(all_q))
    

    #On vérifie que all_q est bien rempli
    print(all_q[0])
    '''

    # On crée les bmin des vendeurs en fonction de chaque q

    for q in all_q: # Pour associer un bmin à  chaque q, on additionne le nombre de 1 dans q puis on divise ce nombre par 96 (sorte de moyenne)
        nombre_de_uns = sum(q[1:])
        bmin = nombre_de_uns / len(q[1:]) 
        bid = (-bmin, q)
        all_bids.append(bid)
        

    ###############################################################################

    # Côté acheteur

    # Une fois le dictionnaire créé, on va l'utiliser pour le Logistic Classifier pour déterminer le bmax des acheteurs

    def classifier(jeu):

        seuil_trouvé = False
        while not seuil_trouvé :
            seuil_trouvé = True

            game_data = donnees_dict[jeu] # Le classifier se lance individuellement pour chaque jeu. La variable jeu est d´terminée aléatoirement avec l'appel de la classe Brands définie plus bas


            # Initialisation d'une liste vide pour stocker les données hours_watched
            hours_watched_data = []

            # Parcours des données du jeu pour récupérer les valers des Hours_watched qui sont celles qu'on a décidé de prendre en compte
            for year, year_data in game_data.items():
                for month, month_data in year_data.items():
                    hours_watched_data.append(month_data['hours_watched'])

            # Conversion de la liste en un tableau numpy
            X = np.array(hours_watched_data, dtype=float).reshape(-1, 1)
            

            # On choisi alors un seuil qui marche pour cette série de valeurs
            max_iterations = 1000
            for _ in range(max_iterations):
                seuil = np.random.randint(np.percentile(X, 25), np.percentile(X, 75)) # On choisit un seuil aléatoire entre le 1er et le 3ème quartile des valeurs de X
                y = np.where(X > seuil, 1, 0).flatten()
                unique_classes = np.unique(y)
                if len(unique_classes) > 1 and np.min(np.bincount(y)) > 1: # On vérifie que les classes sont bien distribuées pour le bon fonctionnement du modèle
                    break
            else:
                # Si le seuil n'est pas trouvé après max_iterations, on recommence
                seuil_trouvé = False
                break 

        if seuil_trouvé : 
            # Code base pour le classifier
            scaler = preprocessing.StandardScaler().fit(X)
            X_scaled = scaler.transform(X)

            # On divise les données en données d'entraînement et de test
            trainX, testX, trainy, testy = train_test_split(X_scaled, y.flatten(), test_size=0.5, random_state=None, stratify=y.flatten()) # Quelle test_size mettre?

            model = LogisticRegression(solver='liblinear',max_iter=500) # Logistic model
            model.fit(trainX, trainy)

            # On prédit les probabilités
            lr_probs = model.predict_proba(testX)
            # On ne garde que les probabilités pour la classe positive
            lr_probs = lr_probs[:, 1]

            error = 0
            for j in range(len(lr_probs)):
                temp = np.round(lr_probs[j])
                if temp != testy[j]:
                    error = error + 1/len(lr_probs)


            # Clacul de bmax

            bmax = 1 - error

            return bmax

    # On définitit maintenant la classe Brands qui va permettre de générer les offres des acheteurs
    class Brands :
        def __init__(self) :
            self.bid = ()
            self.jeu = random.choice(list(donnees_dict.keys())) # On choisit un jeu aléatoire dans le dictionnaire
            
        def generate_bid(self):
            bmax = classifier(self.jeu) # On récupère bmax
            q_acheteur = [self.jeu] + [0 for _ in range(96)] # On initialise q à  [0(x96)] + le nom du jeu
            self.bid = (bmax, q_acheteur)

        # Permet de demander les bons produits (dates disponibles)
        def set_q(self, all_q):
        # On parcourt les q de all_q pour trouver le q correspondant au jeu de l'acheteur
            for q in all_q:
                if q[0] == self.jeu:
                    # Si il y a 1 dans le q du vendeur (mm jeu) alors 50% chance qu'il y ait -1
                    modified_q = [self.jeu] + [random.choice([-1, 0]) if value == 1 else value for value in q[1:]] # Intéressant pour qu'il y est différentes offres
                    self.bid = (self.bid[0], modified_q)
                    return  # On a trouvé le q correspondant, on peut sortir de la boucle


            

    nb_brands = nb_acheteurs # On va modifier cette valeur pour le calcul du surplus

    brands_list = [Brands() for _ in range(nb_brands)]



    # Générer une offre pour chaque marque dans la liste et l'ajouter au marché (all_bids)
    for brand in brands_list:
        brand.generate_bid() 
        brand.set_q(all_q)
        all_bids.append(brand.bid)


    def make_var(i):
        return pulp.LpVariable(f"w{i}", lowBound=0, cat="Integer")

    # Fonction qui filtre les bids pour un certain jeu
    def filter_bid(all_bids, nom_jeu) :
        new_bids = [0]*len(all_bids)
        for i in range(len(all_bids)) :
            if all_bids[i][1][0] == nom_jeu : 
                new_bids[i] = all_bids[i]
        return new_bids # New_bids regroupe les offres du même jeu en gardant les mêmes indices que dans all_bids, bid = 0 si c'est pas le bon jeu

    # Une fois toutes les offres filtrées, on optimise le matching avec offres du même jeu
    def optimization(new_bids) :
        A = len(new_bids)
        b = [0]*A
        q = [0]*A
        # On va remplir b et q en gardant les mêmes indices que dans all_bids
        for i in range(A):
            if new_bids[i] != 0 : # Donc si le vendeur i ne concerne pas le bon jeu, on laisse 0 à l'indice i
                b[i] = new_bids[i][0] 
                q[i] = new_bids[i][1]
        
        prob = pulp.LpProblem("Winner", pulp.LpMaximize)
        wvars = [make_var(i) for i in range(A)]
        prob += pulp.lpSum(b[i]*wvars[i] for i in range(A) if b[i] != 0) # Fonction objectif à maximiser
        # Contrainte sur la quantité de produit pour chaque produit j
        for j in range(1, 97): # Commence au deuxième élément pour omettre le nom du jeu
            prob += pulp.lpSum(q[i][j]*wvars[i] for i in range(A) if q[i] != 0) >= 0 
        # Contrainte sur le nb de bids gagnantes par agent
        for i in range(A):
            prob += wvars[i] <= 1
        pulp.LpStatus[prob.solve()]

        # On peut afficher les variables de décisions, si wi = 1 => l'offre de l'agent i choisie (car on a gardé les mêmes indices i depuis all_bids)
        print("WD = ", pulp.value(prob.objective), "\nVariables de décisions : ", [pulp.value(wvars[i]) for i in range(A)])
        
        winning_bids = [new_bids[i] for i in range(A) if pulp.value(wvars[i]) == 1]
        print("Winning bids:")
        for i, bid in enumerate(winning_bids):
            print(f"  Bid {i+1}: {bid}")
        print("\n")
        return pulp.value(prob.objective)
    
    # On calcule le surplus pour chaque jeu et on les sommes pour surplus global
    surplus = 0
    for game in donnees_dict.keys():
        surplus += optimization(filter_bid(all_bids, game))

    return surplus


def main():
    nb_marques = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
    surplus_results = [simulate_marketplace(n,donnees_dict) for n in nb_marques]

    plt.figure(figsize=(10, 5))
    plt.plot(nb_marques, surplus_results, marker='o')
    plt.title('Market Surplus vs Number of Agents')
    plt.xlabel('Number of Agents')
    plt.ylabel('Surplus')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()