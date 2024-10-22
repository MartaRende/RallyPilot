# Format
Toutes les données suivent le format de colonnes suivant:
speed,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,forward,backward,left,right

- [float] speed: vitesse actuelle du vehicule
- [float] r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15: 15 distances obtenues par les rayons 
- [0 or 1] forward,backward,left,right: directions de la voiture actuellement activées (1) ou non (0) 


# Infos fichiers individuels:
1. TrainingData1: j'ai roulé du mieux que je pouvais sans attention particuliere
2. TrainingData2: j'ai roulé plus prudemment
3. TrainingData3: j'ai essayé de longer les bords de la route
4. TrainingData4: plus de données suivant les instructions des TrainingData 2 et 3
5. TrainingData5: compilation des TrainingData 1 à 4
6. TrainingData6: compilation des TrainingData 3 et 4
7. ============= A partir de ce point, la valeur "forward" est toujours a 1
8. TrainingData7: j'ai roulé partout sur la route 
9. TrainingData8: j'ai roulé plus prudemment
10. TrainingData9: compilation des TrainingData 7 et 8
11. TrainingData10: petit extrait de 2 esquives de barrieres
12. TrainingData11: TrainingData9 avec quelques iteration de TrainingData10
13. TrainingData 12 à 14: plus de données suivant les instructions des TrainingData 8 et 9 