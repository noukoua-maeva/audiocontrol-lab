# Comprendre et présenter AudioControl Lab

## L'idée

On demande au modèle deux sons : une version « basse » et une version « haute »
pour une propriété précise. Par exemple, un son grave puis un son aigu. On mesure
si la propriété évolue dans le sens demandé, puis on écoute pour vérifier que
la mesure correspond à ce qui s'est réellement passé.

Un score correct ne suffit pas : un modèle peut produire un sifflement aigu alors
qu'on lui demandait une flûte. La fréquence a augmenté, mais la consigne complète
n'est pas respectée. C'est pourquoi le projet sépare mesure acoustique et écoute.

## Les fichiers à comprendre

1. `configs/smoke.json` décrit trois paires de textes et les paramètres.
2. `experiment.py` prépare les essais, appelle le modèle et sauvegarde les sons.
3. `metrics.py` calcule le volume numérique, une estimation de hauteur et le rythme.
4. `evaluate.py` compare chaque paire, résume et construit le rapport audio.
5. `notebooks/run_gpu.ipynb` orchestre l'exécution sur un GPU.

## Les notions importantes

**Seed / graine.** Le modèle part d'un bruit aléatoire. Fixer la graine permet de
réutiliser une condition de départ comparable. Plusieurs graines permettent
d'observer la variabilité. Même graine ne signifie pas même son après génération.

**Diffusion.** Un modèle préentraîné transforme progressivement du bruit en une
représentation audio, en tenant compte du texte. Le projet utilise les poids
existants d'AudioLDM2 ; il ne les entraîne pas.

**Guidance.** Un paramètre influence la force avec laquelle le texte oriente la
génération. Ici, on compare trois valeurs, sans supposer que la plus élevée sera
toujours meilleure.

**RMS et dBFS.** La RMS mesure l'amplitude moyenne quadratique du signal numérique.
Doubler l'amplitude augmente la mesure d'environ 6.02 dB. Cela ne suffit pas à
mesurer la sensation de volume d'une personne.

**F0.** La fréquence fondamentale est liée à la hauteur d'un son périodique.
Un bruit de pluie n'a pas nécessairement une F0 pertinente. Le code peut donc
retourner une valeur absente au lieu d'inventer une fréquence.

**Bootstrap.** On rééchantillonne les paires de textes pour observer la variation
d'une statistique. Trois générations d'une même phrase ne représentent pas trois
nouvelles phrases indépendantes. Avec quatre paires, l'incertitude reste grande.

## Ce qui est déjà démontré

La version initiale exécute les tests et analyse six signaux mathématiques connus.
Cette calibration vérifie que les mesures réagissent comme prévu. Les signaux ne
sont ni une génération de pluie ni un résultat d'AudioLDM2. Le notebook GPU reste
à exécuter pour obtenir les premiers vrais résultats du modèle.

## Avant de présenter le projet en entretien

- Exécuter le notebook et écouter toutes les premières sorties.
- Expliquer une paire réussie et une paire ratée à partir des fichiers réels.
- Savoir pourquoi on n'a pas normalisé chaque fichier audio.
- Comprendre comment une mesure peut être trompeuse.
- Distinguer utilisation de PyTorch pour l'inférence et entraînement d'un réseau.
- Expliquer quelles décisions tu as revues, modifiées ou approfondies après la
  génération initiale du code assistée par Codex.

Sur un CV, le statut actuel est **projet en cours : développement d'un outil
d'évaluation de la contrôlabilité audio**. Une fois les expériences réelles
exécutées et comprises, on pourra ajouter leurs résultats vérifiés. Ne pas
annoncer d'amélioration du modèle, de fine-tuning ou de précision sans preuve.
