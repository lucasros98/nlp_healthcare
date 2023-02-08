from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def show_entities(tagger, sentences):
    tagged_sentences = tagger.predict(sentences)
    if tagger.bert_tokenizer is not None:
        word_encoded = tagger.bert_tokenizer(sentences, is_split_into_words=True, truncation=True, 
                                             max_length=tagger.params.bert_max_len).input_ids
        sentences = [[tagger.bert_tokenizer.decode(i) for i in s[1:-1]] for s in word_encoded]
    
    styles = {
        'LOC': 'background-color: #aaffaa; color: black;',
        'PER': 'background-color: #aaaaff; color: black;',
        'ORG': 'background-color: #ff8800; color: black;',
        'MISC': 'background-color: #00ffff; color: black;',

        'disorder': 'background-color: #ff3333; color: white;',
        'drug': 'background-color: #44bbff; color: white;',
        'bodypart': 'background-color: #308227; color: white;'
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


def plot_training(ner_system):
  fig, ax = plt.subplots(1, 2, figsize=(2*6,1*6))
  ax[0].plot(ner_system.history['train_loss']);
  ax[0].set_title('Training loss')
  ax[1].plot(ner_system.history['val_f1']);
  ax[1].set_title('Validation F-score')