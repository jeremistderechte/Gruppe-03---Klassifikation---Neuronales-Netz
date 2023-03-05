# Gruppe-03-Klassifikation-Neuronales-Netz

Willkommen bei unserer Lösung zur Klassifikation mit einem neuronalen Netz.

# Benoetigte Bibliotheken
- scikit-learn
- pandas
- tensorflow (cpu oder gpu)
- numpy
- matplotlib
- pickle
- flask (benoetigt fuer Demo)

# Demo starten
1. Repository clonen
2. Terminal/Shell in Ordner `Demo` öffnen
3. `flask run` eingeben
4. localhost (127.0.0.1) im Webbrowser öffnen (Standardport 5000), wurde auch waehrend der Praesentation gehostet

![image](https://user-images.githubusercontent.com/116145963/220192295-d9bb395b-1fd5-49a6-a7ad-52963156e9ae.png)


5. Daten des Kundens eingeben und auf `Abschicken` klicken, um Ergebnis zu bekommen
6. Neue Seite mit dem Ergebnis sollte sich oeffnen

# Training

## Python mit Keras
![image](https://user-images.githubusercontent.com/116145963/222932073-2b14f502-0747-4552-b063-d72eb22b6d9a.png)


| metric | result |
| --- | --- |
| precision | 0.845 |
| recall | 0.813 |
| accuracy | 0.812 |

## Knime

![Download (3)](https://user-images.githubusercontent.com/116145963/221373882-1bb34dc9-a582-42d1-9bdc-cf388e937099.png)

| metric | result |
| --- | --- |
| precision | 0.707 |
| recall | 0.691 |
| accuracy | 0.717 |


![image](https://user-images.githubusercontent.com/116145963/218550747-5661cb20-cc06-48ac-8460-de1cd9f08ea0.png)


# Evaluation (threshold)

![image](https://user-images.githubusercontent.com/116145963/222956923-cd2d2002-4915-4a76-91bc-59a1141c542d.png)


Man kann mit dem threshold arbeiten, welcher als "Steuerinstrument" dient, damit man auf Kosten von predictions weniger FPs generiert, was für diesen Use Case sinnvoll sein kann.

![threshold](https://user-images.githubusercontent.com/116145963/222957046-0c983988-819f-464d-adae-839688903947.png)


## Confusion Matrix mit threshold
![confusion_matrix_threshold](https://user-images.githubusercontent.com/116145963/222957123-4188d596-1e88-4624-937f-b49e78fb45ac.png)


