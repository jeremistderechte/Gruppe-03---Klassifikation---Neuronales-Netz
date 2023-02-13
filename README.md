# Gruppe-03-Klassifikation-Neuronales-Netz

Willkommen bei unserer Lösung zur Klassifikation mit einem neuronalen Netz.

# Benoetigte Bibliotheken
- scikit-learn
- pandas
- tensorflow (cpu oder gpu)
- numpy
- matplotlib
- flask (benoetigt fuer Demo)

# Demo starten
1. Repository clonen
2. Terminal/Shell in Ordner `Demo` öffnen
3. `flask run` eingeben
4. localhost (127.0.0.1) im Webbrowser öffnen (Standardport 5000)

![image](https://user-images.githubusercontent.com/116145963/218327568-8a1a86ca-c2e5-44f1-b3a4-da0dbd6365f3.png)

5. Daten des Kundens eingeben und auf `Abschicken` klicken, um Ergebnis zu bekommen
6. Neue Seite mit dem Ergebnis sollte sich oeffnen

# Training

## Python mit Keras
![image](https://user-images.githubusercontent.com/116145963/218286288-a62faea0-07c2-412c-8d57-9bc106949f7e.png)

| metric | result |
| --- | --- |
| precision | 0.843 |
| recall | 0.804 |
| accuracy | 0.813 |

## Knime

![image](https://user-images.githubusercontent.com/116145963/218550708-7bb0d4b6-2441-4d47-96cc-a16a674dc128.png)

![image](https://user-images.githubusercontent.com/116145963/218550747-5661cb20-cc06-48ac-8460-de1cd9f08ea0.png)


# Evaluation (threshold)

![image](https://user-images.githubusercontent.com/116145963/218286501-8f00487c-c9f0-45d1-9d61-e0baafade9b4.png)

Man kann mit dem threshold arbeiten, welcher als "Steuerinstrument" dient, damit man auf Kosten von predictions weniger FPs generiert, was für diesen Use Case sinnvoll sein kann.

![threshold_with_perfect_threshold](https://user-images.githubusercontent.com/116145963/218287180-83a469d1-c7fb-4009-a4b7-bc670d3bf075.png)


## Confusion Matrix mit threshold
![threshold_confusion_matrix](https://user-images.githubusercontent.com/116145963/218286477-3d8aed83-c38d-4039-a0ea-61006a32eff8.png)



# Sonstiges
<details><summary>Organisation</summary>
<p>

## Theorie
- 20 Minuten

### Literaturrecherche 
- Hanna hat schon damit angefangen
	- Zeitstrahl für PNN
	- Einordnung von Neuronalen Netzen

## Praxis 
- 20 Minuten

### Knime
- Eike hat damit angefangen

### Python Code
- Jeremy hat damit angefangen (fertig, nur noch refactoring)
- Demo, Praxis Beispiel (fertig)


# Präsentation an sich
- Hanna kann das
- Hanna und Eike?

</p>
</details>



