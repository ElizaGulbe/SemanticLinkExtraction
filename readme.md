This repository contains a solution for hypernymy and synonymy detection problem using machine learning applications. 

Prerequisite : you should have an established connection to Tēzaurs DB (production or a copy). You should create your own .env file with the following parameters to run the files. 

DB_NAME - username
DB_PASSWORD - pasword 
DB_NAME - name of the database 


The link detection process consists mainly of 3 important steps : 

1) Training process
2) Candidate generation
3) Link extraction / detection 

1. Training process.  

The training process contains of 5 steps : 
* Connection to Tēzaurs DB 
* Data embedding 
* Generation of negative dataset examples
* Training process
* Results analysis 

2. Candidate generation. 

To narrow the search space, we have developed c



The training data consists of manually labeled dataset and includes these relations - synonymy, hypernymy, antonymy, holonymy, similar and also. 

To DO - definitions of these links 

The process of extracting positive examples is available under 1_Training process/Dataset/1_positive_example_extraction.py

We use the manually labeled dataset as the basis for the training training dataset. 

Afterwards we embed the data using HPLT embedding. More info about the HPLT is avialable here - https://huggingface.co/HPLT/hplt_bert_base_lv

The embedding process is available at 1_Training process/Dataset/2_embed_positive_examples.py

We extract negative examples based on the positive training dataset. 

To DO : write specifically about the  





Primārais dokuments ir repozitorija readme.md; ja kaut vajadzīgs kas nav tur, tad tas readme satur URL norādi uz to kur ir.
pirmā lieta tur - nodokumentēt prerequisites kam ir jābūt lai to palaistu; ja kaut kādus failus/db jāņem no ārpuses, tad pateikt ko vajag un kur likt vai kur konfigurēt vietu;
otrā lieta tur - instrukcija, kā reproducēt eksperimentus lai dabūtu galvenos mērījumciparus (tas ir grūti, jo tas nozīmē ka vajag arī varēt puslīdz viegli reproducēt)
trešā lieta tur - instrukcijā, kā palaist pārapmācību un kur ko konfigurēt;