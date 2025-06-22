# Potencjalne cechy przykładów do analizy

- theme (Tematyczność) - artykuł porusza temat, który porusza statement (1 - tak, 0 - nie). Jeśli wartość jest równa 0, to wszystkie inne pola są puste.

- genericity (Generyczność) - artykuł mówi ogólnie (np. o systemie wyborczym USA), czy konkretnie o 2000 roku i Florydzie? (1 - ogólnie, 0 - konkretnie).

- verbosity (Redundancja) - artykuł powtarza te same informacje ze statement bez dodawania nowej wiedzy do omawianego tematu (1 - powtarza, 0 - dodaje nowe informacje).

- informativness (Informacyjność) - artykuł uwzględnia informacje, które nie znajdują się w statement ani justification (1 - uwzględnia, 0 - nie uwzględnia)

- truthfulness1 (Prawdziwość 1) - informacje zawarte w artykule są prawdziwe (2 - tak, 1 - częściowo tak, 0 - fałszywe)

- truthfulness2 (Prawdziwość 2) - dodatkowe informacje zawarte w artykule są prawdziwe (2 - tak, 1 - częściowo tak, 0 - fałszywe, PUSTE - nic nie dodaje)

- usefulness (Użyteczność) - artykuł uwzględnia potencjalnie przydatne informacje do klasyfikacji (1 - tak, 0 - nie, -1 - szkodzą, PUSTE - brak dodatkowych informacji po za tym co jest w statement)

- usefull_facts (Przydatne fakty) - przydatne fakty zawarte w artykule, wypunktowane

- reverse (Odwrotność statementu) - Treść artykułu jawnie zaprzecza treści statementu (1 - przekazuje odwrotną tezę, 0 - przekazuje to samo)
