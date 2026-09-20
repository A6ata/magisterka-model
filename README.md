# Aplikacja Streamlit do obrony pracy magisterskiej

Aplikacja-prezentacja do pracy:
**„Wpływ nieliniowej funkcji asymilacji zanieczyszczeń na dynamikę modelu zrównoważonego wzrostu gospodarczego”**.

## Co zawiera

Główna narracja obrony ma 8 sekcji:
1. Problem badawczy i cel pracy
2. Model bazowy
3. Modyfikacja i wyniki analityczne
4. Parametryzacja i scenariusz bazowy
5. Analiza wrażliwości
6. Dynamika przejściowa
7. Wielopunktowość
8. Wnioski i ograniczenia

Sekcja 9 („Symulacje”) jest zapleczem do pytań komisji i pozwala zmieniać parametry oraz warunki początkowe.

## Struktura repozytorium

Wszystkie pliki powinny znajdować się **bezpośrednio w głównym katalogu repozytorium GitHub**:

```text
app.py
analysis.py
model.py
dynamics.py
multiplicity_region.csv.gz
requirements.txt
README.md
```

Nie trzeba tworzyć folderu `data`.

## Uruchomienie lokalne

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Wdrożenie na Streamlit Community Cloud

1. Utwórz repozytorium na GitHubie.
2. Wgraj wszystkie 7 plików wymienionych wyżej do katalogu głównego repozytorium.
3. Wejdź na https://share.streamlit.io i zaloguj się przez GitHub.
4. Wybierz `Create app`.
5. Wskaż repozytorium i branch `main`.
6. Jako `Main file path` wpisz dokładnie:

```text
app.py
```

7. Kliknij `Deploy`.

## Ważne

Plik `multiplicity_region.csv.gz` zawiera jedyną preobliczoną, kosztowniejszą mapę wielopunktowości. Analiza wrażliwości, scenariusz bazowy, punkty stacjonarne oraz pojedyncze trajektorie są wyliczane przez aplikację z kodu modelu.

Jeżeli plik mapy nie zostanie wgrany, aplikacja nie powinna się wywrócić: pokaże ostrzeżenie, a przykład wielopunktowości P̄=400, mNL=0,05 nadal zostanie obliczony bezpośrednio.
