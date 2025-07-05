# Potencjalne cechy przykładów do analizy

- theme (Tematyczność) - artykuł porusza temat, który porusza statement (1 - tak, 0 - nie). Jak zero to inne pola = 0.

- genericity (Generyczność) - artykuł mówi ogólnie o temacie, który porusza statement (1 - tak, 0 - nie)

- specificity (Konkretność) - artykuł mówi konkretnie o temacie, który porusza statement (1 - tak, 0 - nie)

- verbosity (Redundancja) - artykuł powtarza te same informacje ze statement bez dodawania nowej wiedzy do omawianego tematu (1 - tak, 0 - nie).

- additional_info (Dodaje informacje) - artykuł dodaje nowe informacje do omawianego tematu (1 - tak, 0 - nie)

- informativness (Informacyjność) - artykuł uwzględnia informacje, które nie znajdują się w statement ani justification. Interesują nas takie artykuły, które zawierają dodatkową wiedzę, która poprawia klasyfikację. (1 - tak, 0 - nie)

- truth1_true (Prawdziwość1 tak) - informacje zawarte w artykule są prawdziwe (1 - tak, 0 - nie)

- truth1_partial (Prawdziwość1 częściowa) - informacje zawarte w artykule są przynajmniej częściowo prawdziwe (1 - tak, 0 - nie)

- truth1_false (Prawdziwość1 żadna) - informacje zawarte w artykule są fałszywe (1 - tak, 0 - nie)

- truth1_unknown (Prawdziwość1 nieznana) - trudno stwierdzić prawdziwość informacji zawartych w artykule (1 - tak, 0 - nie)

- truth2_true (Prawdziwość2 tak) - dodatkowe informacje zawarte w artykule są prawdziwe (1 - tak, 0 - nie)

- truth2_partial (Prawdziwość2 częściowa) - dodatkowe informacje zawarte w artykule są częściowo prawdziwe (1 - tak, 0 - nie)

- truth2_false (Prawdziwość2 żadna) - dodatkowe informacje zawarte w artykule są fałszywe (1 - tak, 0 - nie)

- usefulness (Użyteczność) - artykuł uwzględnia potencjalnie przydatne informacje do klasyfikacji (1 - tak, 0 - nie)

- harmfulness (Szkodliwość) - artykuł uwzględnia potencjalnie szkodliwe informacje do klasyfikacji (1 - tak, 0 - nie)

- usefull_facts (Przydatne fakty) - przydatne fakty zawarte w artykule, wypunktowane

- reverse (Odwrotność statementu) - Treść artykułu jawnie zaprzecza treści statementu (1 - tak, 0 - nie)

- complaisance (Zgodność statementu) - Treść artykułu jest zgodna z tezą zawartą w treści statement (1 - tak, 0 - nie)

- neutral (Neutralność statementu) - Treść artykułu ani jawnie nie jest zgodna ani nie zaprzecza treści statementu (1 - tak, 0 - nie)
