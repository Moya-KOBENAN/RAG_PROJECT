# Pipeline d'ingestion RAG

Cette première partie du projet transforme des documents PDF, TXT ou DOCX en
vecteurs et les enregistre dans un index FAISS local.

## Prérequis

- Python 3.12
- Une connexion Internet lors du premier lancement, afin de télécharger le
  modèle `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

## Installation

Depuis le dossier `RAG` :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

`requirements.txt` contient les dépendances directes. Le fichier
`requirements-lock.txt` verrouille aussi leurs dépendances indirectes et sert
à reproduire l'environnement validé.

## Données

Déposer les documents dans `data/documents/`. Les formats acceptés sont PDF,
TXT et DOCX. Le nom du fichier source et le numéro de page sont conservés dans
les métadonnées des passages indexés.

Vérifier que les documents sont identiques au corpus validé :

```bash
cd data && shasum -a 256 -c SHA256SUMS && cd ..
```

## Création de l'index

```bash
python app.py
```

Si le corpus et les paramètres n'ont pas changé, le pipeline conserve l'index
existant. Pour imposer une reconstruction :

```bash
python app.py --force
```

La configuration peut être adaptée sans modifier le code :

| Variable | Valeur par défaut |
|---|---|
| `RAG_DOCUMENTS_DIR` | `data/documents` |
| `RAG_INDEX_DIR` | `index/faiss_index` |
| `RAG_CHUNK_SIZE` | `500` |
| `RAG_CHUNK_OVERLAP` | `100` |

Avec le corpus actuel, le résultat attendu est :

- 2 documents PDF et 8 pages chargées ;
- 20 passages créés avec une taille de 500 caractères et un chevauchement de 100 ;
- un index écrit dans `index/faiss_index/`.

Le nouvel index est contrôlé puis publié de façon atomique : en cas d'échec,
l'index précédent reste disponible. Un fichier `manifest.json` enregistre les
empreintes des documents, les paramètres, la révision du modèle et les nombres
de pages et de passages. Les fichiers source ne sont jamais modifiés.

## Tests

Les tests n'utilisent ni le réseau ni le modèle d'embeddings :

```bash
python -m unittest discover -s tests -v
```

Valider séparément l'intégrité et l'actualité de l'index :

```bash
python validate.py
```

## Automatisation avec Airflow

Airflow utilise un environnement séparé afin de ne pas modifier les
dépendances du RAG :

```bash
python3.12 -m venv .venv-airflow
source .venv-airflow/bin/activate
python -m pip install -r requirements-airflow-lock.txt
export AIRFLOW_HOME="$PWD/airflow"
airflow db migrate
airflow standalone
```

Le DAG `rag_ingestion_pipeline` exécute trois contrôles successifs :

```text
check_environment → run_pipeline → validate_index
```

Il est planifié chaque jour à 02:00 dans le fuseau `Africa/Abidjan`. Sa
planification peut être changée avant le démarrage d'Airflow :

```bash
export RAG_AIRFLOW_SCHEDULE="0 6 * * *"
```

Les chemins ne sont pas liés à cette machine. Si nécessaire, définir
`RAG_PROJECT_DIR` et `RAG_PYTHON`. Le DAG limite les exécutions concurrentes à
une, applique un délai maximal d'une heure et effectue deux nouvelles
tentatives après un échec.

## Structure

```text
app.py                 orchestration de l'ingestion
validate.py            contrôle autonome de l'index publié
airflow/dags/           automatisation et supervision
data/documents/        corpus documentaire
src/loader.py          chargement des documents
src/chunker.py         découpage en passages
src/embedding.py       modèle d'embeddings multilingue
src/config.py          configuration centralisée
src/manifest.py        traçabilité et détection des changements
src/pipeline.py        orchestration réutilisable par Airflow
src/vector_store.py    création et sauvegarde de l'index FAISS
index/faiss_index/     index généré
tests/                 tests automatiques
```

## API de recherche

L'API effectue une recherche vectorielle locale et retourne les passages
pertinents avec leurs sources :

Après le démarrage du service, ouvrir l'interface interactive Swagger dans le
navigateur :

- interface complète : <http://localhost:8000/docs> ;
- formulaire de la route de recherche :
  <http://localhost:8000/docs#/default/query_query_post>.

Dans Swagger, ouvrir `POST /query`, cliquer sur **Try it out**, saisir la
question puis cliquer sur **Execute**.

La même route peut être appelée directement depuis un terminal :

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Quels sont les principaux indicateurs ?"}'
```

Le corps de la requête contient uniquement `question`. Le serveur retourne
automatiquement les trois passages les plus proches.

Les routes de supervision sont :

- `GET /health/live` : vérifie que le processus API répond ;
- `GET /health/ready` : vérifie que l'index FAISS est présent et valide ;
- `GET /index/status` : retourne le manifeste de l'index.

## Mise en production locale reproductible avec Docker

La même image et les mêmes paramètres peuvent être utilisés sur toute machine
équipée de Docker. Le service reste accessible uniquement depuis la machine
hôte sur `localhost:8000` et n'est pas publié sur Internet.

Créer la configuration locale à partir du modèle :

```bash
cp .env.example .env
```

Le fichier `.env` est exclu de Git et du contexte Docker. Aucune clé API n'est
requise, car le port est lié exclusivement à l'interface locale `127.0.0.1`.

Construire, initialiser l'index et démarrer le service en arrière-plan :

```bash
docker compose up -d --build
docker compose ps
```

Le service éphémère `rag-init` vérifie le corpus et construit automatiquement
l'index lorsqu'il est absent ou obsolète. L'API ne démarre qu'après la réussite
de cette étape. Si l'index est déjà à jour, il est conservé.

Vérifier la disponibilité :

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

Interroger l'API :

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Quels sont les principaux indicateurs ?"}'
```

Consulter les journaux et arrêter proprement le service :

```bash
docker compose logs -f rag-api
docker compose down
```

Le conteneur utilise un utilisateur non privilégié, un système de fichiers en
lecture seule, aucune capacité Linux additionnelle et des limites CPU/mémoire.
Le corpus est monté en lecture seule et seul le dossier de l'index est
inscriptible. Le modèle d'embedding est intégré à l'image afin que les
documents et les questions restent sur la machine et que les recherches ne
dépendent pas du réseau après la construction.

Pour reproduire le service sur une autre machine, copier le projet avec le
dossier `data/documents`, installer Docker, créer `.env` depuis
`.env.example`, puis exécuter `docker compose up -d --build`. Aucun Python ni
modèle supplémentaire ne doit être installé sur la machine hôte.
