# Projet GPU partie Stencil 

## Description

Ce code implémente un calcul de type stencil sur un tableau à deux dimensions. Le calcul stencil est utilisé dans les simulations scientifiques pour modéliser des phénomènes décrits par des équations aux dérivées partielles. Ce programme effectue des itérations sur un tableau, appliquant à chaque valeur et à son voisinage un motif de calcul spécifique.

L'objectif du projet est d'améliorer les performances de ce code de calcul, en utilisant **SIMD AVX2** basé sur la vectorisation et **StarPU** basé sur la parallélisation par division de tâches.

## Versions améliorées
Le but de ce projet est de coder plusieurs versions améliorées de la fonction `naive_stencil_func()`.
- **Amélioration de la fonction :**  Pour mieux traiter les autres parties et optimiser notre fonction, on commence par modifier mathématiquement le parcours de nos valeur en jouant sur le reste du division euclidienne de l'indice globale noté Ig et x.
- **Version StarPU :** On utilise la librairie StarPU pour gérer la parallélisation et l'optimisation des calculs de stencil en répartissant les tâches de calcul sur différents coeurs de processeur.
- **Version SIMD_AVX2 :** Vectorise les opérations sur les vecteurs faîtes dans notre fonction optimisée, traitant plusieurs données simultanément par instruction.
- **Version StarPU+AVX2 :** Combine les avantages de StarPU et SIMD_AVX2, en utilisant la parallélisation de StarPU pour répartir efficacement les tâches entre les coeurs du processeur, tout en appliquant la vectorisation AVX2.

## Compilation

1) Connectz-vous au cluster miriel et copier le dossier dans un répertoire.
2) charger les modules nécessaires pour compiler le code avec la commande `source load_modules`.
3) mettez make pour créer l'exécutable.



## Execution

Pour exécuter le programme, utilisez la commande ./stencil_fusion pour le compiler avec les valeurs par défauts(starpu avec 12 tâches et verifier les résultats).
Pour tester le code pour plusieurs valeurs sans avoir à l'exécuter à chaque fois, ajoutez les options :

\```

- `--mesh-width M` : Largeur du tableau.
- `--mesh-height N` : Hauteur du tableau.
- `--nb-iterations I` : Nombre d'itérations à effectuer.
- `--nb-repeat NB_REPEAT` : Nombre de répétition du calcul à effectuer.
- `--version V` : Choix de la version du calcul (`naive`, `starpu`, `nvx`, `starpu_nvx`).
- `--nb_parts P` : Nombre de tâches pour StarPU.
- `--verification` : Activer (`1`) ou désactiver (`0`) la vérification des calculs.
- `--initial-mesh <zero|random>` : Pour commencer soit avec un mesh zéro ou le générer aléatoirement.
- `--output` : Pour enregistrer la matrice de sortie.
- `--verbose` : fournir plus de détails lors de l'éxecution.

## Exemples

Pour exécuter le programme avec une largeur de 2000, une hauteur de 1000, 10 itérations en mode `starpu_nvx` et en verifiant le resultat avec `naive_stencil_func()`:

\"" ./stencil_fusion --nb-iterations 10 --mesh-width 4000 --mesh-height 1000 --verification 1 --version starpu_nvx --taskpart 1
\```

## Principaux résultats obtenus
On calcule le temps d'exécution pour **mesh_width = 5050**, **mesh_height = 3062**, **Nombre de partitions de StarPU = 24** et avec 10 répétitions.




## Auteurs

- **NAIM Mouna**  - mnaim@bordeaux-inp.fr
- **REDOUANE Mohamed Youssef**   - mredouane@enseirb-matmeca.fr
