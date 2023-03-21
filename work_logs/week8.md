# Week 6

## Meeting notes (21/3)

Data augmentation
- Kom på nåt sätt så att datamängderna blir lika att jämföra
- Vilken metod är bäst?
- Hur mycket ny data är bäst?
- Bara kolla 25% original + 75% augmenterad? (blir det bättre än 100% original)

Rapport:
- Innehållsförteckning
    - Ha inte bara en subsection, bara ligga i texten? Eller slå ihop två sections?
    - Kanske flytta Transfer Learning under BERT
- Intro
    - Första stycket, hur det relaterar till vår rapport, vad kommer vi göra
    - Ge överblick på en högre nivå - lösa det här problemet
- Problem
    - börja med aim och gå vidare där ifrån
    - Vi vill titta på klinisk data, det medför en hel del problem...
- Aim
    - Lägg till "clinical sector" i sista stycket
    - objectives => contributions
        - Istället identifiera det bästa sättet för att hantera svensk text data, typ peka på den bästa approachen för denna settingen
        - Göra FÖR ATT ...
- I början på avsnitt referera till målet (typ related work, Theory), varför är det viktigt
- Modellerna (theory)
    - Beskriv om den är tränad på engelsk eller svensk text
    - För de engelska modellerna, kanske beskriv att de kan köras på svensk text om de översätts (referera framåt till att vi kommer göra det)
    - Mer detalj på BERT figuren
- Methods
    - Igen referera tillbaka till målet
    - Nämns inte att vi anävnder subsets av datan osv
    - Det som är viktigt i metoden ska highlightas i början, sedan kommer mer i detalj
- Evaluation
    - ska den kanske vara under Methods? 
    - Gå ingen lite noggrannare, verkade lite svårt att förstå i nuläget
    - Varför har vi med text-similarity metrics (varför viktigt att ha med?)
- Discussion
    - Formulera rubriker efter ett antal frågor
    - "Vilken är den bästa approachen för att hantera svensk, klinisk textdata?"

# Preliminary plan for the week