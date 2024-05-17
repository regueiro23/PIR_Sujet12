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


nom_fichier = './Twitch_game_data.csv' # Nom du fichier CSV contenant les donnÃ©es

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
        
        # Ajout de chaque valeur Ã   la liste appropriÃ©e dans le dictionnaire
        donnees[game_name][year][month] = {'hours_watched': hours_watched, 'hours_streamed': hours_streamed, 'peak_viewers': peak_viewers, 'peak_channels': peak_channels, 'streamers': streamers, 'avg_viewers': avg_viewers}

# On a un dictionnaire oÃ¹ chaque clÃ© est un nom de jeu, chaque sous-clÃ© est une annÃ©e, chaque sous-sous-clÃ© est un mois, et la valeur est un autre dictionnaire avec 'hours_watched' et 'hours_streamed'

# On dÃ©cide d'enlever les jeux du dictionnaire qui moins de 4 entrÃ©es de donnÃ©es pour faciliter le fonctionnement du Logistic Classifier
# On crÃ©e une liste de jeux Ã   enlever
jeux_a_enlever = [game for game, data in donnees.items() if len(data) < 4]

# On enlÃ¨ve les jeux de la liste des jeux Ã   enlever
for game in jeux_a_enlever:
    del donnees[game]

# Maintenant le dictionnaire ne contient que des jeux avec 4 entrÃ©es de donnÃ©es ou plus

# On converti le dictionnaire en un dictionnaire rÃ©gulier
def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d

donnees_dict = defaultdict_to_dict(donnees)

'''
# DiffÃ©rents tests Ã  faire
# Affichage de tous les Ã©lÃ©ments du dictionnaire
for game_name, game_data in donnees_dict.items():
    print(f"{game_name}: {game_data}\n")

    
# Ou afficher seulement les premiers Ã©lÃ©ments de chaque jeu
for game_name, game_data in list(donnees_dict.items())[:5]:
    print(f"{game_name}: {game_data}\n")

    
# Affichage du premier Ã©lÃ©ment du dictionnaire (League of Legends) afin de faire des tests avec les q retournÃ©s par remplissage_q()
first_key = next(iter(donnees_dict))
first_value = donnees_dict[first_key]


print(f"First key: {first_key}")
print(f"First value: {first_value}")
'''

def simulate_marketplace(nb_acheteurs,donnees_dict):

    

    ###############################################################################

    # CÃ´tÃ© vendeur
    all_bids = [] # all_bids va contenir toutes les offres du marchÃ©

    q_size = 96 # Taille du vecteur q <=> nombre de types de donnÃ©es diffÃ©rents = 8*12 = 96 (une entrÃ©e pour chaque mois entre 2016 et 2023)


    # Remplissage de all_q avec les vecteurs q possibles pour chaque jeu en fonction des donnÃ©es prÃ©sentÃ©es dans la dataset
    def remplissage_q():

        all_q = [] # all_q contient tous les vecteurs q possible (avec 0 et 1) 
        for game_name, game_data in donnees_dict.items():
            q = [game_name] + [0]*q_size # On ajoute le nom du jeu au vecteur q
            for year, year_data in game_data.items():
                for month, value in year_data.items():
                    # On remplit q en fonction des donnÃ©es prÃ©sentes par annÃ©e et mois
                    if year == "2016":
                        k = 0
                    elif year == "2017":
                        k = 12
                    elif year == "2018":
                        k = 24
                    elif year == "2019":
                        k = 36
                    elif year == "2020":
                        k = 48
                    elif year == "2021":
                        k = 60
                    elif year == "2022":
                        k = 72
                    elif year == "2023":
                        k = 84
                    
                    j = 0
                    if month == "01":
                        j = 0
                    elif month == "02":
                        j = 1
                    elif month == "03":
                        j = 2
                    elif month == "04":
                        j = 3
                    elif month == "05":
                        j = 4
                    elif month == "06":
                        j = 5
                    elif month == "07":
                        j = 6
                    elif month == "08":
                        j = 7
                    elif month == "09":
                        j = 8
                    elif month == "10":
                        j = 9
                    elif month == "11":
                        j = 10
                    elif month == "12":
                        j = 11
                    
                    q[j+k+1]= 1 # +1 pour Ã©viter de modifier la premiÃ¨re entrÃ©e de q qui correspond au nom du jeu
            all_q.append(q)
        return all_q

    all_q = remplissage_q()

    '''
    # DiffÃ©rents tests Ã  faire
    # On vÃ©rifie que tous les vecteurs q sont dans all_q 
    print(len(all_q))
    

    #On vÃ©rifie que all_q est bien rempli
    print(all_q[0])
    '''

    # On crÃ©e les bmin des vendeurs en fonction de chaque q

    for q in all_q: # Pour associer un bmin Ã   chaque q, on additionne le nombre de 1 dans q puis on divise ce nombre par 96 (sorte de moyenne)
        nombre_de_uns = sum(q[1:])
        bmin = nombre_de_uns / len(q[1:]) 
        bid = (-bmin, q)
        all_bids.append(bid)
        

    ###############################################################################

    # CÃ´tÃ© acheteur

    # Une fois le dictionnaire crÃ©Ã©, on va l'utiliser pour le Logistic Classifier pour dÃ©terminer le bmax des acheteurs

    def classifier(jeu):

        seuil_trouvÃ© = False
        while not seuil_trouvÃ© :
            seuil_trouvÃ© = True

            game_data = donnees_dict[jeu] # Le classifier se lance individuellement pour chaque jeu. La variable jeu est dÂ´terminÃ©e alÃ©atoirement avec l'appel de la classe Brands dÃ©finie plus bas


            # Initialisation d'une liste vide pour stocker les donnÃ©es hours_watched
            hours_watched_data = []

            # Parcours des donnÃ©es du jeu pour rÃ©cupÃ©rer les valers des Hours_watched qui sont celles qu'on a dÃ©cidÃ© de prendre en compte
            for year, year_data in game_data.items():
                for month, month_data in year_data.items():
                    hours_watched_data.append(month_data['hours_watched'])

            # Conversion de la liste en un tableau numpy
            X = np.array(hours_watched_data, dtype=float).reshape(-1, 1)
            

            # On choisi alors un seuil qui marche pour cette sÃ©rie de valeurs
            max_iterations = 1000
            for _ in range(max_iterations):
                seuil = np.random.randint(np.percentile(X, 25), np.percentile(X, 75)) # On choisit un seuil alÃ©atoire entre le 1er et le 3Ã¨me quartile des valeurs de X
                y = np.where(X > seuil, 1, 0).flatten()
                unique_classes = np.unique(y)
                if len(unique_classes) > 1 and np.min(np.bincount(y)) > 1: # On vÃ©rifie que les classes sont bien distribuÃ©es pour le bon fonctionnement du modÃ¨le
                    break
            else:
                # Si le seuil n'est pas trouvÃ© aprÃ¨s max_iterations, on recommence
                seuil_trouvÃ© = False
                break 

        if seuil_trouvÃ© : 
            # Code base pour le classifier
            scaler = preprocessing.StandardScaler().fit(X)
            X_scaled = scaler.transform(X)

            # On divise les donnÃ©es en donnÃ©es d'entraÃ®nement et de test
            trainX, testX, trainy, testy = train_test_split(X_scaled, y.flatten(), test_size=0.5, random_state=None, stratify=y.flatten()) # Quelle test_size mettre?

            model = LogisticRegression(solver='liblinear',max_iter=500) # Logistic model
            model.fit(trainX, trainy)

            # On prÃ©dit les probabilitÃ©s
            lr_probs = model.predict_proba(testX)
            # On ne garde que les probabilitÃ©s pour la classe positive
            lr_probs = lr_probs[:, 1]

            error = 0
            for j in range(len(lr_probs)):
                temp = np.round(lr_probs[j])
                if temp != testy[j]:
                    error = error + 1/len(lr_probs)


            # Clacul de bmax

            bmax = 1 - error

            return bmax

    # On dÃ©finitit maintenant la classe Brands qui va permettre de gÃ©nÃ©rer les offres des acheteurs
    class Brands :
        def __init__(self) :
            self.bid = ()
            self.jeu = random.choice(list(donnees_dict.keys())) # on choisit un jeu alÃ©atoire dans le dictionnaire
            
        def generate_bid(self):
            bmax = classifier(self.jeu) # On rÃ©cupÃ¨re bmax
            q_acheteur = [self.jeu] + [0 for _ in range(96)] # On initialise q Ã   [0(x96)] + le nom du jeu
            self.bid = (bmax, q_acheteur)

        #permet de demander les bons produits (dates disponibles)
        def set_q(self, all_q):
        # Look for the q which has the correct game name in all_q
            for q in all_q:
                if q[0] == self.jeu:
                    #Si il y a 1 dans le q du vendeur (mm jeu) alors 50% chance qu'il y ait -1
                    modified_q = [self.jeu] + [random.choice([-1, 0]) if value == 1 else value for value in q[1:]] # intÃ©ressant pour qu'il y est diffÃ©rentes offres
                    self.bid = (self.bid[0], modified_q)
                    return  # Exit the method once the q is found and updated


            

    nb_brands = nb_acheteurs # On va modifier cette valeur pour le calcul du surplus

    brands_list = [Brands() for _ in range(nb_brands)]



    # GÃ©nÃ©rer une offre pour chaque marque dans la liste et l'ajouter au marchÃ© (all_bids)
    for brand in brands_list:
        brand.generate_bid() 
        brand.set_q(all_q)
        all_bids.append(brand.bid)


    def make_var(i):
        return pulp.LpVariable(f"w{i}", lowBound=0, cat="Integer")

    # fonction qui filtre les bids pour un certain jeu
    def filter_bid(all_bids, nom_jeu) :
        new_bids = [0]*len(all_bids)
        for i in range(len(all_bids)) :
            if all_bids[i][1][0] == nom_jeu : 
                new_bids[i] = all_bids[i]
        return new_bids #new_bids regroupe les offres du mÃªme jeu en gardant les mÃªmes indices que dans all_bids, bid = 0 si c'est pas le bon jeu

    # une fois toutes les offres filtrÃ©es, on optimise le matching avec offres du mÃªme jeu
    def optimization(new_bids) :
        A = len(new_bids)
        b = [0]*A
        q = [0]*A
        # on va remplir b et q en gardant les mÃªmes indices que dans all_bids
        for i in range(A):
            if new_bids[i] != 0 : # donc si le vendeur i ne concerne pas le bon jeu, on laisse 0 Ã  l'indice i
                b[i] = new_bids[i][0] 
                q[i] = new_bids[i][1]
        
        prob = pulp.LpProblem("Winner", pulp.LpMaximize)
        wvars = [make_var(i) for i in range(A)]
        prob += pulp.lpSum(b[i]*wvars[i] for i in range(A) if b[i] != 0) # fonction objectif Ã   maximiser
        # contrainte sur la quantitÃ© de produit pour chaque produit j
        for j in range(1, 97):  # commence au deuxiÃ¨me Ã©lÃ©ment pour omettre nom du jeu
            prob += pulp.lpSum(q[i][j]*wvars[i] for i in range(A) if q[i] != 0) >= 0
        # contrainte sur le nb de bids gagnantes par agent
        for i in range(A):
            prob += wvars[i] <= 1
        pulp.LpStatus[prob.solve()]

        # on peut afficher les variables de dÃ©cisions, si wi = 1 => l'offre de l'agent i est choisie (car on a gardÃ© les mÃªmes indices i depuis all_bids)
        print("WD = ", pulp.value(prob.objective), "\nVariables de dÃ©cisions : ", [pulp.value(wvars[i]) for i in range(A)])
        
        winning_bids = [new_bids[i] for i in range(A) if pulp.value(wvars[i]) == 1]
        print("Winning bids:")
        for i, bid in enumerate(winning_bids):
            print(f"  Bid {i+1}: {bid}")
        print("\n")
        return pulp.value(prob.objective)
    
    # on calcul le surplus pour chaque jeu et on les sommes pour surplus global
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
