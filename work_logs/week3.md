# Week 3

## Premilinary plan for the week

- Start running jobs on the cluster
- Finetune models on the data set (on the cluster)

## Notes from weekly meeting
- Make a section about clinical health data?
- Maybe put Related Work chapter before discussion
- How can we change the data set to our advantage (language model)?
- Having different data sets that test different things, like one testing new data (new "first names")
- Maybe sum our findings about the data set - give feedback?
- Check out other NER papers, what mapping / tagging approach do they use?

## Work done this week

This week we started working with Alvis. In the beginning of the week we ran some simple scripts to learn. Then we moved on the the actual data set and finetuning models.

We also improved the translation script to be able to handle the data set better, and now we can run it on the cluster, which is much faster.

One thing we observed was that most of the models achieved a very high F1 score. We believe that this could be because of two reasons:
- The data is very simple for the models to learn.
- Our tagging approach may not be accurate when calculating the F1 score.

SweDeClinBERT achieved the highest F1 score, but the other models also performed very well.

## Work plan for next week

- Explore the tagging approach -> perhaps we can improve it to achieve a more accurate F1 score.
- Improve our scripts.
- Finetuning BioBERT and M-BERT on English text data (translated from Swedish).
- Writing on the report.
