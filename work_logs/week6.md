# Week 6

## Meeting notes (6/3)

- Threshold på precision / recall ?
- är olika 25 splits mer eller mindre unikt? Kanske
- Är det storleken på datasetet som spelar roll eller unikheten?
    - Enligt forskning är det mer värt med kvalitet över kvantitet
    - Kanske ta fram två testset som är väldigt lika / unika
- Projektpitch: hur löser man bra NER system för clinical data när du har väldigt lik / olik data
    - Behöver inte vara en mening, lite grenar men ska ändå hänga samman
    - Vad är det vi löser egentligen? NER på ett dataset som är städat exakt som vi gjort?
        - Kanske: hur behöver man processera kliniska dataset?
- Kommer data augmentation presetera bättre eller sämre om ursprungsdatan är lik / olik
- Vilka resultat skulle vi vilja presentera i slutändan? Vad för tabeller/ plots etc..
    - Vill vi ha ett bättre resultat på det givna testsetet eller definera om testsetet etc.
    - Lovisa: hon tycker att iv behöver ha med pitchen (vad är det vi ska lösa) - hur kopplar allt vi gör till pitchen?
- T5 modellen: processering; ta bort alla rader som inte slutar med '. , ! ? "'
    - Ta bort sidor om de innehåller färre än 3 meningar / mindre än 5 ord

- I resultat: dessa processerings steg gjordes / behövs för att nå detta resultat
- Spara ner den nya datan

- Plots: istället hur många Locations exempel (träningssamples) har modellen sett? på x-axeln
- I varje split: stratified sampling: man samplar typ random men har nån form av vikt
    - man bevarar andelen labels i varje split (om 100% har 10 Locations kommer 10% ha samma andel)
    - Vanligt att göra med test / träningsdatan

- Istället för antalet epoker: titta istället på hur många batches har modellen sett oavsett data size
    - Vi tränar varje modell på exakt 1h, eller exakt såhär många samples ska modellen få se
    - Finns verktyg för att lösa det
    - Ge en träningsbudget, men stoppa tidigare ifall man ser att lossen går upp
    - Vi gav dem en max compute budget (när den konvergerade i tränings loss)
    - Träna dem 10 gånger och ta ett medelvärde (först när vi vet allt om datan etc.)

- Ska vi välja ut de "bästa" modellerna? Inte än kanske, men senare?
    - Vad vi orkar med / hinner 
    - Lovisa: 2 modellval: ta den bästa greedy eller ta hänsyn till vilka modeller som finns tillgängliga för folk att avända
    - Fundera över ifall det finns nåt intressant att titta på just denna modellen?

- Nu har vi visat prestanda resultat: men visa nu också antecdotal evidence / results
    - Vad hände? Taggade den rätt? När den översatte, vad taggade då? 
    - Komplettera plottarna

## Premilinary plan for the week

- Ta fram en ny test-dataset med endast unika entiteter, delvis baserat på Sahlgrenskas data, SCB-statistik osv.
  - Sedan träna om modellerna och jämför med det tidigare resultaten - Hur presterar modellerna på unika entities?
- Börja med data augmentation och implementera back translation.
- Skapa graf över olika similarity metrics för datan och augmentation techniques.