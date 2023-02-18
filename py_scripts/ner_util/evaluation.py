from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
plt.style.use('seaborn')

text_examples_sv = [
    'ssk Lisa försöker att få tag i BIB Jonas Berg. Vid behov, informera pat fru när plats finns ledig på Sophiahemmet. Förhoppningsvis innan 20230221',
    'Åtgärd	Dietist hör av sig och kommer boka om tiden till 12/7 kl 14. Besök kommer ske på avd 314',
    'Besöksorsak	Kontakt med dr Kayed, tel nr 0711442373. Pat är 82 år, längd 175 cm och mår illa efter måltider.',
]
text_examples_en = [
    "ssk Lisa tries to get hold of BIB Jonas Berg. If necessary, inform the pat wife when the place is available at Sophiahemmet. Hopefully before 20230221",
    "Action Dietist will be in touch and will reschedule the time to 12/7 at 2 p.m. Visits will be made on avd 314",
    "Contact with Dr. Kayed, tel. 0711442373. Pat is 82 years old, length 175 cm and feels nauseous after meals."
]

def print_examples(ner_system, lang):
    if(lang == "sv"):
        for i, sentence in enumerate(text_examples_sv):
            print(f"\nSV example {i + 1}")
            show_entities(ner_system, [sentence.lower().split()])
            print("")
    else:
        for i, sentence in enumerate(text_examples_en):
            print(f"\nEN example {i + 1}")
            show_entities(ner_system, [sentence.lower().split()])
            print("")

def show_entities(ner_system, sentence):
    tagging_system = ner_system.params.tagging_scheme
    tagged_sentence = ner_system.predict(sentence)
    if ner_system.bert_tokenizer is not None:
        word_encoded = ner_system.bert_tokenizer(sentence, is_split_into_words=True, truncation=True, 
                                             max_length=ner_system.params.bert_max_len).input_ids
        sentence = [[ner_system.bert_tokenizer.decode(i) for i in s[1:-1]] for s in word_encoded]
    
    labels = ['First_Name', 'Last_Name', 'Phone_Number', 'Age', 'Full_Date', 'Date_Part', 'Health_Care_Unit', 'Location']

    print('All token tags:')
    for tokens, tags in zip(sentence, tagged_sentence):
        output = ''
        for token, tag in zip(tokens, tags):
            if tag in labels:
                output += f"<{tag}> {token} </{tag}> "
            else:
                output += f"{token} "
        print(output)

    print('Combined token tags:')
    for tokens, tags in zip(sentence, tagged_sentence):
        output = ''
        last_tag = ''
        last_token = ''
        for token, tag in zip(tokens, tags):
            # print tags
            if last_tag in labels and not token.startswith("##"):
                output += f" </{last_tag}>"
            if tag in labels and not token.startswith("##"):
                 output += f" <{tag}>"

            # print tokens
            if token.startswith("##"):
                output += f"{token}"
            else:
                output += f" {token}"

            # update last token and tag
            last_token = token
            last_tag = tag

        # add final closing tag if final token was labeled
        if last_tag in labels:
            output += f" </{last_tag}> "
        print(output)
            

def show_entities_html(tagger, sentences):
    tagged_sentences = tagger.predict(sentences)
    if tagger.bert_tokenizer is not None:
        word_encoded = tagger.bert_tokenizer(sentences, is_split_into_words=True, truncation=True, 
                                             max_length=tagger.params.bert_max_len).input_ids
        sentences = [[tagger.bert_tokenizer.decode(i) for i in s[1:-1]] for s in word_encoded]
    
    labels = ['First_Name', 'Last_Name', 'Phone_Number', 'Age', 'Full_Date', 'Date_Part', 'Health_Care_Unit', 'Location']

    styles = {
        'First_Name': 'background-color: #aaffaa; color: black;',
        'Last_Name': 'background-color: #aaaaff; color: black;',
        'Phone_Number': 'background-color: #ff8800; color: black;',
        'Age': 'background-color: #00ffff; color: black;',
        'Full_Date': 'background-color: #ff3333; color: white;',
        'Date_Part': 'background-color: #44bbff; color: white;',
        'Health_Care_Unit': 'background-color: #308227; color: white;',
        'Location': 'background-color: #308227; color: white;'

    }
    content = ['<div style="font-size:150%; line-height: 150%;">']

    for tokens, tags in zip(sentences, tagged_sentences):
        content.append('<div>')
        current_entity = None
        for token, tag in zip(tokens, tags):
            if tag[0] not in ['B', 'I']:
                if current_entity:
                    content.append('</b>')
                    current_entity = None
                content.append(' ')
            elif tag[0] == 'B':
                if current_entity:
                    content.append('</b>')
                content.append(' ')
                current_entity = tag[2:]
                content.append(f'<b style="{styles[current_entity]} border-radius: 3px; padding: 3px;">')
                content.append(f'<sup style=font-size:small;><tt>{current_entity}</tt></sup> ')

            else:
                entity = tag[2:]
                if entity == current_entity:
                    content.append(' ')
                elif current_entity is None:
                    content.append(' ')
                    content.append('<sup style=font-size:small;><tt>[ERROR]</tt></sup> ')
                    content.append(f'<b style="{styles[entity]} border-radius: 3px; padding: 3px;">')
                    content.append(f'<sup style=font-size:small;><tt>{entity}</tt></sup> ')
                else:
                    content.append('</b>')
                    content.append(' ')
                    content.append('<sup style=font-size:small;><tt>[ERROR]</tt></sup> ')
                    content.append(f'<b style="{styles[entity]} border-radius: 3px; padding: 3px;">')
                    content.append(f'<sup style=font-size:small;><tt>{entity}</tt></sup> ')
                current_entity = entity
            content.append(token)
        if current_entity:
            content.append('</b>')
        content.append('</div>')
    content.append('</div>')    
    html = ''.join(content).strip()
    display(HTML(html))
    with open("entities.png", "wb") as png:
        png.write(HTML(html))


def plot_training(ner_system):
  fig, ax = plt.subplots(1, 2, figsize=(2*6,1*6))
  ax[0].plot(ner_system.history['train_loss'])
  ax[0].set_title('Training loss')
  ax[1].plot(ner_system.history['val_f1'])
  ax[1].set_title('Validation F-score')
  plt.savefig('training_plot2.png')