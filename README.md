# Klasyfikator z pracy inżynierskiej

Związany projekt: https://github.com/jegor377/liar_plus_analysis

## Struktura katalogów

- `data` - katalog zawierający przeanalizowany i przetworzony zbiór danych LIAR PLUS
- `data/normalized` - katalog, który zawiera znormalizowany zbiór danych LIAR PLUS
- `notebooks` - pliki notatników wykorzystane do wytrenowania różnych architektur modeli
- `analysis` - pliki notatników wykorzystane do analizy wyników trenowania modeli
- `analysis/base model analysis` - pliki analizy bazowego modelu w rozdziale 5
- `analysis/input methods analysis` - pliki analizy reprezentacji wejściowych w rozdziale 5
- `analysis/final experiment analysis` - pliki analizy modelu SM i SMA z ostatecznego eksperymentu z rozdziału 7
- `analysis/models eval` - pliki ewaluacji jakości różnych modeli (wszystkie z rozdziału 7)

## Istotne pliki

- `Eval SM vs SMA.ipynb` - notatnik wykorzystany do ewaluacji zbioru testowego przy pomocy modelu SM i SMA, zebranie metryk SM_pred, SMA_pred, SM_prob, SMA_prob, label_num, SM_highest_prob, SMA_highest_prob i zapisanie do pliku `sm_vs_sma_dataset.csv`
- `SM vs SMA comparison.ipynb` - notatnik wykorzystany do dalszej analizy wyników `sm_vs_sma_dataset.csv` i wyodrębniający po 30 przykładów dla każdego modelu, do dalszej analizy (opisane w rozdziale 7.3)
- `wyniki eksperymentu 1.ods` - wyniki eksperymentu reprezentacji tekstu z rozdziału 5
- `analysis/final experiment analysis/SM_correct_SMA_incorrect.ods` - 30 przykładów zbioru SM_correct_SMA_incorrect opisane w rozdziale 7.3
- `analysis/final experiment analysis/SM_incorrect_SMA_correct.ods` - 30 przykładów zbioru SM_incorrect_SMA_correct opisane w rozdziale 7.3
- `wyniki ostatecznego eksperymentu.ods` - wyniki trenowania modeli z rozdziału 7

- `normalize_dataset.py` - skrypt do normalizacji zbioru danych
- `utils.py` - paczka do obsługi zapisu/odczytu modeli oraz transmisji na nasz serwer

- `demo_gemma.py` - generator artykułów do programu demonstracyjnego.
- `demo_pipelines.py` - generator dodatkowych kolumn metadanych, których nie ma w oryginalnym zbiorze LIAR PLUS.
- `demo_utils.py` - pliki pomocnicze i definicja modelu PyTorch.
- `demo.py` - główny plik programu demonstracyjnego.
