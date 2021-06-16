# Valuing-players

Aquest repositori conté els fitxers necessaris per a obtenir els ratings de tots els jugadors compressos en el dataset d'StatsBomb, excepte el propi dataset que es pot trobar
en el següent enllaç: https://github.com/statsbomb/open-data

El projecte s'estructura en tres mòduls diferents i que s'executen independentment per acabar obtenint el rating dels jugadors:
* data_extraction
* action_values
* valuing_players

# Data_extraction

Extreu els events de cada partit present dins el dataset de StatsBomb i els converteix en format d'acció per a poder valorar-les. 
Els guarda dins de la carpeta /dataframes/ amb l'id del partit com a nom de fitxer i en format .pkl
Per dur a terme l'extracció de dades s'ha d'executar l'arxiu data_extraction.py de la següent forma:

* py data_extraction.py

Per extreure totes les accions

* py data_extraction.py shots

Per extreure només els xuts

# Action_values

Aplica la fórmula per valorar cada acció i assignar un valor total a cada una d'aquestes.
Per a dur a terme la valoració de les accions s'ha d'executar l'arxiu action_values.py de la següent forma:

* py action_values.py

# Valuing_players

Adjudica a cada jugador la valoració de les seves accions i en retorna el rating final. Quan s'executa el fitxer corresponent es despelga un menú que deixa triar 
les següents opcions per les quals es volen obtenir els jugadors:

* Competició
* Gènere, si s'han triat totes les competicions
* Porter o jugadors de camp
* Nombre de jugadors que es volen obtenir

El fitxer s'executa de la següent forma:

* py valuing_players.py
