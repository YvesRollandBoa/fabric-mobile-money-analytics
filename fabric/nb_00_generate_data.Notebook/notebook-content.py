# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "7e88a803-31b1-419e-98cf-23781e2529dc",
# META       "default_lakehouse_name": "lh_mobile_money",
# META       "default_lakehouse_workspace_id": "1524f296-0a79-4523-82cb-29ff62fffbe0",
# META       "known_lakehouses": [
# META         {
# META           "id": "7e88a803-31b1-419e-98cf-23781e2529dc"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # nb_00_generate_data — Générateur de données mobile money (synthétiques)
# 
# Simule l'activité d'un opérateur de mobile money fictif en Côte d'Ivoire (montants en XOF).
# 
# **Sorties (Lakehouse `lh_mobile_money`, zone `Files/landing/`)** — format « système source » :
# - `referentiels/` : zones, clients, agents, marchands (instantanés CSV complets)
# - `transactions/aaaa/mm/jj/transactions_aaaammjj.csv` : un fichier par jour, comme un export quotidien d'opérateur
# - `verite_fraude/fraud_labels.csv` : étiquettes de fraude, **séparées** des transactions.
#   Un vrai flux opérateur ne dit pas quelles transactions sont frauduleuses : ce fichier ne sert
#   qu'à mesurer la détection, jamais à la construire.
# 
# **Hypothèses de simulation** (détaillées dans le README) : saisonnalité (fin de mois, Korité,
# Tabaski, rentrée, fêtes), croissance de la base clients, adoption progressive de l'application,
# grille de frais fictive inspirée des pratiques du marché, 4 scénarios de fraude (~0,3 %).
# 
# **Prérequis** : Attacher `lh_mobile_money` comme Lakehouse **par défaut** du notebook avant d'exécuter.


# CELL ********************

# ---------------------------------------------------------------- Paramètres
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEED = 42
DATE_DEBUT = date(2025, 1, 1)
DATE_FIN = date.today() - timedelta(days=1)       # jusqu'à hier : le flux temps réel prendra le relais
N_TRANSACTIONS = 3_000_000                         # volume cible (hors fraude injectée)
N_CLIENTS = 50_000
N_AGENTS = 1_500
N_MARCHANDS = 2_000
TAUX_ECHEC = 0.025

# Chemin local du Lakehouse par défaut (API fichiers) ; repli local pour tester hors Fabric
BASE = "/lakehouse/default/Files/landing" if os.path.exists("/lakehouse/default/Files") else "./landing"

rng = np.random.default_rng(SEED)
print(f"Période : {DATE_DEBUT} → {DATE_FIN} | sortie : {BASE}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---------------------------------------------------------------- Référentiel géographique
# Population : RGPH 2021, résultats définitifs (INS Côte d'Ivoire), dernier recensement disponible —
# communes d'Abidjan, sous-préfectures pour l'intérieur. Coordonnées : centre approximatif de la zone.
ZONES = [
    # code, nom, région / district, lat, lon, population 2021
    ("ABJ-ABO", "Abobo",        "District d'Abidjan",       5.416, -4.020, 1_340_083),
    ("ABJ-ADJ", "Adjamé",       "District d'Abidjan",       5.367, -4.023,   340_892),
    ("ABJ-ATT", "Attécoubé",    "District d'Abidjan",       5.335, -4.040,   313_135),
    ("ABJ-COC", "Cocody",       "District d'Abidjan",       5.354, -3.987,   692_583),
    ("ABJ-KOU", "Koumassi",     "District d'Abidjan",       5.297, -3.955,   412_282),
    ("ABJ-MAR", "Marcory",      "District d'Abidjan",       5.300, -3.983,   214_061),
    ("ABJ-PLA", "Plateau",      "District d'Abidjan",       5.323, -4.022,     7_186),
    ("ABJ-PBO", "Port-Bouët",   "District d'Abidjan",       5.255, -3.929,   618_795),
    ("ABJ-TRE", "Treichville",  "District d'Abidjan",       5.292, -4.009,   106_552),
    ("ABJ-YOP", "Yopougon",     "District d'Abidjan",       5.336, -4.088, 1_571_065),
    ("BKE",     "Bouaké",       "Gbêkê",                    7.690, -5.030,   832_371),
    ("YAM",     "Yamoussoukro", "District de Yamoussoukro", 6.827, -5.289,   340_234),
    ("SPE",     "San-Pédro",    "San-Pédro",                4.748, -6.636,   390_654),
    ("KHO",     "Korhogo",      "Poro",                     9.458, -5.629,   440_926),
    ("DAL",     "Daloa",        "Haut-Sassandra",           6.877, -6.450,   421_879),
    ("MAN",     "Man",          "Tonkpi",                   7.412, -7.554,   241_969),
    ("GAG",     "Gagnoa",       "Gôh",                      6.131, -5.951,   277_044),
    ("ABG",     "Abengourou",   "Indénié-Djuablin",         6.729, -3.496,   164_424),
]
zones = pd.DataFrame(ZONES, columns=["zone_code", "zone_nom", "region", "latitude", "longitude", "population_2021"])
zones["est_abidjan"] = zones["zone_code"].str.startswith("ABJ")

# Probabilité de tirer chaque zone, proportionnelle à sa population (clients, agents et marchands)
p_zone = (zones["population_2021"] / zones["population_2021"].sum()).to_numpy()

print(zones.assign(part=p_zone.round(3))[["zone_nom", "population_2021", "part"]].to_string(index=False))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---------------------------------------------------------------- Clients
# Inscriptions : un stock avant 2025, puis une croissance continue jusqu'à la fin de période.
n_anciens = int(N_CLIENTS * 0.55)
jours_avant = (DATE_DEBUT - date(2022, 1, 1)).days
jours_periode = (DATE_FIN - DATE_DEBUT).days
inscription = np.concatenate([
    np.datetime64("2022-01-01") + rng.integers(0, jours_avant, n_anciens).astype("timedelta64[D]"),
    np.datetime64(DATE_DEBUT) + rng.integers(0, jours_periode, N_CLIENTS - n_anciens).astype("timedelta64[D]"),
])
inscription.sort()   # identifiants attribués dans l'ordre d'inscription, comme dans un vrai système

segment = rng.choice(["particulier", "commercant"], N_CLIENTS, p=[0.92, 0.08])
clients = pd.DataFrame({
    "client_id": [f"C{i:06d}" for i in range(1, N_CLIENTS + 1)],
    "zone_code": rng.choice(zones["zone_code"], N_CLIENTS, p=p_zone),
    "tranche_age": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], N_CLIENTS,
                              p=[0.24, 0.34, 0.22, 0.12, 0.08]),
    "segment": segment,
    "niveau_kyc": rng.choice([1, 2, 3], N_CLIENTS, p=[0.35, 0.50, 0.15]),
    "date_inscription": inscription,
})
# Intensité d'usage (non exportée) : quelques gros utilisateurs, beaucoup d'occasionnels
intensite = rng.gamma(shape=0.9, scale=1.0, size=N_CLIENTS) * np.where(segment == "commercant", 3.0, 1.0)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---------------------------------------------------------------- Agents & marchands
agents = pd.DataFrame({
    "agent_id": [f"A{i:05d}" for i in range(1, N_AGENTS + 1)],
    "zone_code": rng.choice(zones["zone_code"], N_AGENTS, p=p_zone),
    "type_point": rng.choice(["kiosque", "boutique", "station-service", "supermarche"], N_AGENTS,
                             p=[0.55, 0.30, 0.08, 0.07]),
    # Réseau ouvert avant la période : aucune transaction ne peut précéder l'ouverture d'un point
    "date_ouverture": np.datetime64("2019-01-01")
                      + rng.integers(0, (DATE_DEBUT - date(2019, 1, 1)).days, N_AGENTS).astype("timedelta64[D]"),
})

CATEGORIES = ["supermarche", "carburant", "restauration", "telecom", "ecole", "pharmacie"]
n_commerces = N_MARCHANDS - 3
commerces = pd.DataFrame({
    "marchand_id": [f"M{i:05d}" for i in range(1, n_commerces + 1)],
    "nom_commercial": [f"Marchand {i:05d}" for i in range(1, n_commerces + 1)],
    "categorie": rng.choice(CATEGORIES, n_commerces, p=[0.23, 0.12, 0.26, 0.10, 0.13, 0.16]),
    "zone_code": rng.choice(zones["zone_code"], n_commerces, p=p_zone),
})
# Les factures vont à quelques grands facturiers nationaux, pas à des centaines de marchands
facturiers = pd.DataFrame({
    "marchand_id": [f"M{i:05d}" for i in range(n_commerces + 1, N_MARCHANDS + 1)],
    "nom_commercial": ["Facturier électricité", "Facturier eau", "Facturier TV-Internet"],
    "categorie": "facture",
    "zone_code": "ABJ-PLA",            # siège au Plateau ; le paiement se fait partout
})
marchands = pd.concat([commerces, facturiers], ignore_index=True)
print(f"{len(clients):,} clients | {len(agents):,} agents | {len(marchands):,} marchands")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---------------------------------------------------------------- Volume quotidien (saisonnalité)
jours = pd.date_range(DATE_DEBUT, DATE_FIN, freq="D")
t = np.arange(len(jours)) / max(len(jours) - 1, 1)

facteur = 1.0 + 0.40 * t                                                     # croissance de l'activité
facteur *= np.array([1.00, 0.97, 0.98, 1.00, 1.10, 1.06, 0.80])[jours.dayofweek]  # lun → dim
fin_de_mois = (jours.day >= 25) | (jours.day <= 4)                           # salaires
facteur *= np.where(fin_de_mois, 1.30, 1.0)
facteur *= np.where((jours.month == 9) & (jours.day <= 20), 1.15, 1.0)      # rentrée scolaire
facteur *= np.where((jours.month == 12) & (jours.day >= 15), 1.25, 1.0)     # fêtes de fin d'année

# Fêtes religieuses (dates approximatives) : pic les jours qui précèdent
EVENEMENTS = {"2025-03-30": 1.35, "2025-06-06": 1.55, "2026-03-20": 1.35, "2026-05-27": 1.55}  # Korité, Tabaski
for d, f in EVENEMENTS.items():
    ecart = (jours - pd.Timestamp(d)).days
    facteur *= np.where((ecart >= -4) & (ecart <= 1), f, 1.0)

n_jour = rng.poisson(N_TRANSACTIONS * facteur / facteur.sum())
N = int(n_jour.sum())
print(f"{N:,} transactions normales générées ({N / len(jours):,.0f} / jour en moyenne)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---------------------------------------------------------------- Horodatage
HEURES = np.array([0.3, 0.2, 0.1, 0.1, 0.2, 0.6, 2.0, 4.5, 6.5, 6.8, 6.5, 6.2,
                   6.8, 6.5, 5.8, 5.6, 6.0, 6.8, 7.2, 6.4, 4.8, 3.2, 1.8, 0.9])
idx_jour = np.repeat(np.arange(len(jours)), n_jour)
horodatage = (jours.values[idx_jour]
              + rng.choice(24, N, p=HEURES / HEURES.sum()).astype("timedelta64[h]")
              + rng.integers(0, 3600, N).astype("timedelta64[s]"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---------------------------------------------------------------- Émetteur : uniquement des clients déjà inscrits
# Clients triés par date d'inscription + poids cumulés : un tirage pondéré parmi les k premiers
# revient à tirer parmi ceux inscrits avant le jour de la transaction (vectorisé, sans boucle).
ordre = np.argsort(clients["date_inscription"].to_numpy())
insc_tries = clients["date_inscription"].to_numpy().astype("datetime64[ns]")[ordre]
poids_cum = np.cumsum(intensite[ordre])
k = np.searchsorted(insc_tries, jours.values[idx_jour], side="right")       # nb de clients inscrits
emetteur = ordre[np.searchsorted(poids_cum, rng.random(N) * poids_cum[k - 1])]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---------------------------------------------------------------- Type d'opération, contrepartie, canal
TYPES = ["depot", "retrait", "transfert", "paiement_marchand", "paiement_facture", "achat_credit"]
type_op = rng.choice(TYPES, N, p=[0.21, 0.19, 0.28, 0.13, 0.06, 0.13])

# Agents : 80 % dans la zone du client, sinon n'importe où
zone_client = clients["zone_code"].to_numpy()[emetteur]
agents_par_zone = {z: g.to_numpy() for z, g in agents.groupby("zone_code")["agent_id"]}
agent_hors_zone = rng.choice(agents["agent_id"], N)
agent_dans_zone = np.empty(N, dtype=object)
for z, ids in agents_par_zone.items():
    m = zone_client == z
    agent_dans_zone[m] = rng.choice(ids, m.sum())
agent = np.where(rng.random(N) < 0.8, agent_dans_zone, agent_hors_zone)

# Marchands : factures → catégorie dédiée ; en septembre, une partie des paiements va aux écoles
ids_factures = marchands.loc[marchands["categorie"] == "facture", "marchand_id"].to_numpy()
ids_ecoles = marchands.loc[marchands["categorie"] == "ecole", "marchand_id"].to_numpy()
ids_autres = marchands.loc[~marchands["categorie"].isin(["facture", "ecole"]), "marchand_id"].to_numpy()
mois = pd.DatetimeIndex(horodatage).month.to_numpy()
p_ecole = np.where(mois == 9, 0.35, 0.06)
marchand = np.where(rng.random(N) < p_ecole, rng.choice(ids_ecoles, N), rng.choice(ids_autres, N))
marchand = np.where(type_op == "paiement_facture", rng.choice(ids_factures, N), marchand)

# Destinataire d'un transfert : tiré uniformément parmi les k clients déjà inscrits (même logique que l'émetteur)
destinataire = clients["client_id"].to_numpy()[ordre[(rng.random(N) * k).astype(int)]]

# Canal : dépôts/retraits chez un agent ; sinon l'application progresse de 35 % à 55 % sur la période
t_tx = t[idx_jour]
canal = np.where(rng.random(N) < 0.35 + 0.20 * t_tx, "application", "ussd")
canal = np.where(np.isin(type_op, ["depot", "retrait"]), "agent", canal)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---------------------------------------------------------------- Montants (XOF)
def lognormal_arrondi(mediane, sigma, pas, n):
    return np.maximum(pas, np.round(rng.lognormal(np.log(mediane), sigma, n) / pas) * pas)

montant = np.zeros(N)
REGLES = {  # médiane, dispersion, arrondi
    "depot": (10_000, 0.90, 500), "retrait": (15_000, 0.85, 500), "transfert": (7_500, 1.00, 100),
    "paiement_marchand": (5_000, 0.95, 50), "paiement_facture": (12_000, 0.70, 5),
}
for typ, (med, sig, pas) in REGLES.items():
    m = type_op == typ
    montant[m] = lognormal_arrondi(med, sig, pas, m.sum())
m = type_op == "achat_credit"
montant[m] = rng.choice([100, 200, 500, 1_000, 2_000, 5_000], m.sum(), p=[0.10, 0.18, 0.30, 0.25, 0.12, 0.05])
m = (type_op == "paiement_marchand") & np.isin(marchand, ids_ecoles)
montant[m] = rng.choice([25_000, 50_000, 75_000, 100_000, 150_000], m.sum())     # frais de scolarité

# Plafond par transaction selon le niveau KYC (fictif)
plafond = np.array([0, 200_000, 1_000_000, 2_000_000])[clients["niveau_kyc"].to_numpy()[emetteur]]
montant = np.minimum(montant, plafond)

tx = pd.DataFrame({
    "horodatage": horodatage, "client_id": clients["client_id"].to_numpy()[emetteur],
    "type_operation": type_op, "montant": montant, "canal": canal,
    "agent_id": np.where(np.isin(type_op, ["depot", "retrait"]), agent, None),
    "marchand_id": np.where(np.isin(type_op, ["paiement_marchand", "paiement_facture"]), marchand, None),
    "destinataire_client_id": np.where(type_op == "transfert", destinataire, None),
    "scenario_fraude": None,
})
tx = tx[~((tx["type_operation"] == "transfert") & (tx["client_id"] == tx["destinataire_client_id"]))]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---------------------------------------------------------------- Injection de fraude (~0,3 %)
clients_ids = clients["client_id"].to_numpy()
agents_ids = agents["agent_id"].to_numpy()
deb_ns, fin_ns = np.datetime64(DATE_DEBUT), np.datetime64(DATE_FIN) + np.timedelta64(1, "D")

def instants(n, heures=None):
    """n instants aléatoires dans la période (optionnellement restreints à certaines heures)."""
    j = rng.integers(0, (fin_ns - deb_ns).astype(int), n).astype("timedelta64[D]")
    h = rng.choice(heures if heures is not None else np.arange(7, 22), n).astype("timedelta64[h]")
    return (deb_ns + j + h + rng.integers(0, 3600, n).astype("timedelta64[s]")).astype("datetime64[ns]")

def clients_inscrits(instants_tx):
    """Un client tiré uniformément parmi ceux déjà inscrits à chaque instant."""
    k = np.searchsorted(insc_tries, instants_tx, side="right")
    return clients_ids[ordre[(rng.random(len(instants_tx)) * k).astype(int)]]

fraudes = []

# 1) Compte mule : transfert d'une victime vers une mule, puis retrait quasi immédiat par la mule
mules = rng.choice(clients.loc[(clients["niveau_kyc"] == 1)
                               & (clients["date_inscription"] < np.datetime64(DATE_DEBUT)), "client_id"], 60, replace=False)
n1 = 1_500
t1 = instants(n1)
mule = rng.choice(mules, n1)
mt1 = np.round(rng.uniform(100_000, 200_000, n1) / 1_000) * 1_000
fraudes.append(pd.DataFrame({"horodatage": t1, "client_id": clients_inscrits(t1), "type_operation": "transfert",
                             "montant": mt1, "canal": "application", "agent_id": None, "marchand_id": None,
                             "destinataire_client_id": mule, "scenario_fraude": "compte_mule"}))
fraudes.append(pd.DataFrame({"horodatage": t1 + rng.integers(120, 900, n1).astype("timedelta64[s]"),
                             "client_id": mule, "type_operation": "retrait", "montant": mt1 - 1_000, "canal": "agent",
                             "agent_id": rng.choice(agents_ids, n1), "marchand_id": None,
                             "destinataire_client_id": None, "scenario_fraude": "compte_mule"}))

# 2) Rafale : 8 à 15 transferts en moins de 10 minutes vers des destinataires différents
n_rafales = 180
tailles = rng.integers(8, 16, n_rafales)
debuts = instants(n_rafales)
t0 = np.repeat(debuts, tailles)
fraudes.append(pd.DataFrame({
    "horodatage": t0 + rng.integers(0, 600, tailles.sum()).astype("timedelta64[s]"),
    "client_id": np.repeat(clients_inscrits(debuts), tailles), "type_operation": "transfert",
    "montant": np.round(rng.uniform(20_000, 50_000, tailles.sum()) / 500) * 500, "canal": "application",
    "agent_id": None, "marchand_id": None, "destinataire_client_id": clients_inscrits(t0),
    "scenario_fraude": "rafale"}))

# 3) Montant inhabituel : transactions existantes dont le montant est multiplié (×20 à ×40)
cible = tx.index[tx["type_operation"].isin(["transfert", "retrait"]).to_numpy()]
choisis = rng.choice(cible, 2_000, replace=False)
plafond_client = (tx.loc[choisis, "client_id"].map(clients.set_index("client_id")["niveau_kyc"])
                  .map({1: 200_000, 2: 1_000_000, 3: 2_000_000}).to_numpy())
tx.loc[choisis, "montant"] = np.minimum(np.round(tx.loc[choisis, "montant"].to_numpy() * rng.uniform(20, 40, 2_000), -3),
                                        plafond_client)   # inhabituel pour le client, mais sous son plafond
tx.loc[choisis, "scenario_fraude"] = "montant_inhabituel"

# 4) Agent anormal : retraits nocturnes de montants ronds chez 15 agents
n4 = 1_800
t4 = instants(n4, heures=np.array([0, 1, 2, 3, 4]))
fraudes.append(pd.DataFrame({
    "horodatage": t4, "client_id": clients_inscrits(t4),
    "type_operation": "retrait", "montant": rng.choice([100_000, 150_000, 200_000], n4), "canal": "agent",
    "agent_id": rng.choice(rng.choice(agents_ids, 15, replace=False), n4), "marchand_id": None,
    "destinataire_client_id": None, "scenario_fraude": "agent_anormal"}))

tx = pd.concat([tx, *fraudes], ignore_index=True)

# Garde-fou : ni l'émetteur ni le destinataire ne transactent avant leur inscription (doit valoir 0)
date_insc = clients.set_index("client_id")["date_inscription"]
avant = (tx["horodatage"] < tx["client_id"].map(date_insc)).sum()
avant_dest = (tx["horodatage"].dt.normalize() < tx["destinataire_client_id"].map(date_insc)).sum()
assert avant == 0 and avant_dest == 0, f"{avant} émetteurs, {avant_dest} destinataires avant inscription"
tx = tx[tx["horodatage"] < pd.Timestamp(DATE_FIN) + pd.Timedelta(days=1)]   # le retrait d'une mule peut déborder

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---------------------------------------------------------------- Statut, frais, identifiants
tx = tx.sort_values("horodatage", ignore_index=True)
n = len(tx)
p_echec = np.where(tx["canal"] == "ussd", TAUX_ECHEC * 1.4, TAUX_ECHEC)
tx["statut"] = np.where((rng.random(n) < p_echec) & tx["scenario_fraude"].isna(), "echouee", "reussie")

# Grille de frais fictive : dépôt, achat de crédit et facture gratuits pour le client
mt = tx["montant"].to_numpy()
frais_retrait = np.select([mt <= 5_000, mt <= 20_000, mt <= 50_000, mt <= 100_000, mt <= 200_000],
                          [100, 250, 500, 1_000, 1_750], default=np.ceil(mt * 0.01))
frais = np.select([tx["type_operation"] == "retrait", tx["type_operation"] == "transfert"],
                  [frais_retrait, np.maximum(25, np.round(mt * 0.01))], default=0)
commission = np.where(tx["type_operation"] == "paiement_marchand", np.round(mt * 0.01), 0)  # payée par le marchand
reussie = tx["statut"].to_numpy() == "reussie"
tx["frais_client"] = np.where(reussie, frais, 0).astype(int)
tx["commission_marchand"] = np.where(reussie, commission, 0).astype(int)
tx["montant"] = tx["montant"].astype(int)
tx.insert(0, "transaction_id", [f"TX{i:010d}" for i in range(1, n + 1)])

print(f"{n:,} transactions au total | fraude : {tx['scenario_fraude'].notna().mean():.3%}")
print(tx["scenario_fraude"].value_counts().to_string())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---------------------------------------------------------------- Données « sales » (qualité réaliste de la source)
# Une vraie source n'est jamais propre. On injecte quelques défauts connus, hors fraude, pour que le
# nettoyage Silver et les tests dbt aient quelque chose à détecter. Chaque défaut est journalisé à part.
propres = tx.index[tx["scenario_fraude"].isna()].to_numpy()
tirage = rng.permutation(propres)                       # ensembles disjoints : un défaut par transaction
DEFAUTS = {"doublon": 3_000, "montant_manquant": 1_500, "client_inconnu": 600,
           "canal_mal_formate": 1_500, "arrivee_tardive": 3_000}
lots, debut = {}, 0
for defaut, nb in DEFAUTS.items():
    lots[defaut] = tirage[debut:debut + nb]
    debut += nb

# Montant manquant : entier « nullable » pour garder des entiers ailleurs (pas de 10500.0)
tx["montant"] = tx["montant"].astype("Int64")
tx.loc[lots["montant_manquant"], "montant"] = pd.NA
# Client inconnu : identifiant absent du référentiel clients
tx.loc[lots["client_inconnu"], "client_id"] = [f"C9{i:05d}" for i in range(len(lots["client_inconnu"]))]
# Canal mal formaté : majuscules ou espace parasite
tx.loc[lots["canal_mal_formate"], "canal"] = (tx.loc[lots["canal_mal_formate"], "canal"].str.upper() + " ")
# Arrivée tardive : la transaction est livrée dans le fichier du lendemain
tx["date_fichier"] = tx["horodatage"].dt.normalize()
tardives = lots["arrivee_tardive"]
tx.loc[tardives, "date_fichier"] = (tx.loc[tardives, "date_fichier"] + pd.Timedelta(days=1)).clip(upper=pd.Timestamp(DATE_FIN))
# Doublon : la même transaction envoyée deux fois (même identifiant)
tx = pd.concat([tx, tx.loc[lots["doublon"]]], ignore_index=True)

anomalies = pd.concat([pd.DataFrame({"transaction_id": tx.loc[ids, "transaction_id"].to_numpy(), "anomalie": d})
                       for d, ids in lots.items()], ignore_index=True)
print(f"{len(anomalies):,} anomalies injectées ({len(anomalies) / len(tx):.2%})")
print(anomalies["anomalie"].value_counts().to_string())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---------------------------------------------------------------- Écriture en zone landing
os.makedirs(f"{BASE}/referentiels", exist_ok=True)
os.makedirs(f"{BASE}/verite_fraude", exist_ok=True)
os.makedirs(f"{BASE}/verite_qualite", exist_ok=True)
zones.to_csv(f"{BASE}/referentiels/zones.csv", index=False)
clients.to_csv(f"{BASE}/referentiels/clients.csv", index=False)
agents.to_csv(f"{BASE}/referentiels/agents.csv", index=False)
marchands.to_csv(f"{BASE}/referentiels/marchands.csv", index=False)

# Vérité terrain : hors du flux de transactions
(tx.loc[tx["scenario_fraude"].notna(), ["transaction_id", "scenario_fraude"]]
   .drop_duplicates().to_csv(f"{BASE}/verite_fraude/fraud_labels.csv", index=False))
anomalies.to_csv(f"{BASE}/verite_qualite/anomalies.csv", index=False)   # journal des défauts injectés

colonnes = ["transaction_id", "horodatage", "client_id", "type_operation", "montant", "frais_client",
            "commission_marchand", "canal", "statut", "agent_id", "marchand_id", "destinataire_client_id"]
jour = tx["date_fichier"].dt.date                       # date de livraison (≠ date de transaction si tardive)
for d, groupe in tx[colonnes].groupby(jour):
    dossier = f"{BASE}/transactions/{d:%Y/%m/%d}"
    os.makedirs(dossier, exist_ok=True)
    groupe.to_csv(f"{dossier}/transactions_{d:%Y%m%d}.csv", index=False)

print(f"Écrit : {jour.nunique()} fichiers quotidiens + référentiels + vérités terrain dans {BASE}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
