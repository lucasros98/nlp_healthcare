#exploing masking with BERT
from transformers import pipeline

#load the model
old = "ssk Lisa försöker att få tag i BIB Jonas Berg. Vid behov, informera pat fru när plats finns ledig på Sophiahemmet. Förhoppningsvis innan 20230221"
text = "ssk Lisa försöker att få [MASK] i BIB Jonas Berg. Vid [MASK], [MASK] pat [MASK] när plats finns ledig på Sophiahemmet. [MASK] innan 20230221"

nlp = pipeline('fill-mask', model='KB/bert-base-swedish-cased')

#run the model
output = nlp(text)

#combine the masked words in the text
for i in range(len(output)):
    # get the best score
    for j in range(len(output[i])):
        if output[i][j] is not None and output[i][j]["token_str"] is not None:
            word = output[i][j]["token_str"]
            text = text.replace('[MASK]', word, 1)
            break

print("Old text:", old, "\n")
print("New text:", text)

