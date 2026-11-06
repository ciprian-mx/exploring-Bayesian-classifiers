# Explorarea Algoritmilor Bayesieni: Naive Bayes și TAN

Acest proiect conține implementarea de la zero (from scratch) a doi clasificatori Bayesieni: **Naive Bayes** și **Tree Augmented Naive Bayes (TAN)**. Scopul proiectului este prezicerea viralității unui cântec pe baza unor caracteristici muzicale continue (precum *energy*, *danceability* etc.), folosind distribuții Gaussiene.

## Structura Proiectului

Proiectul este organizat astfel:

* **`Dataset/`**: Conține seturile de date inițiale și procesate (`.csv`), scriptul de încărcare a datelor (`data_loader.py`) și notebook-urile de Jupyter folosite pentru curățarea datelor și calculul informației mutuale.
* **`src/`**: Conține implementările manuale ale algoritmilor:
  * `NaiveBayesManual.py`: Implementarea clasificatorului Naive Bayes.
  * `TreeAugmentedNBManual.py`: Implementarea clasificatorului TAN (cu arbore de dependențe bazat pe informația mutuală condiționată).
* **`Sklearn implementation/`**: Conține un notebook Jupyter (`NaiveBayesianClassifier.ipynb`) cu implementarea algoritmului folosind biblioteca `scikit-learn`, utilizat pentru compararea și validarea rezultatelor.
* **`main.py`**: Punctul de intrare (entry point) al aplicației. Acest script încarcă datele curățate, antrenează modelele implementate manual, rulează predicțiile pe datele de test și afișează acuratețea și matricea de confuzie.

## Cerințe (Prerequisites)

Pentru a rula acest proiect, asigurați-vă că aveți instalat **Python 3.8+**. Puteți instala toate dependențele necesare folosind fișierul `requirements.txt`.

## Instalare

1. Clonați acest repository sau descărcați fișierul ZIP și extrageți-l.
2. Deschideți un terminal în folderul rădăcină al proiectului.
3. Este recomandată crearea unui mediu virtual (opțional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Pentru Linux/Mac
   venv\Scripts\activate     # Pentru Windows
   ```
4. Instalați pachetele necesare:
    ```bash
    pip install -r requirements.txt

## Rulare

Pentru a vedea rezultatele finale și comparația dintre modele, rulați scriptul principal din terminal:

    ```bash
    python main.py

Dacă doriți să explorați etapele de curățare a datelor sau varianta implementată cu scikit-learn, puteți deschide fișierele .ipynb folosind Jupyter Notebook sau un editor compatibil (ex: VS Code).


