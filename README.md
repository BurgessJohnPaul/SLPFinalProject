## Data

Reviews were obtained from [myleott.com](https://myleott.com/op-spam.html) and split into four sets of txt files.

There are 20 reviews for 20 hotels for each class (400 reviews), making 1600 reviews in total.
```
data/positive/truthful    Truthful reviews from TripAdvisor
data/positive/deceptive   Deceptive reviews from MTturk
data/negative/truthful    Truthful reviews from Web
data/negative/deceptive   Deceptive reviews from MTurk
```

Files are named: `[truthtfulness]_[hotel_name]_[number].txt`, e.g. `t_hilton_1.txt`.

Downloading GloVe embeddings:

The GloVe file used for the project was too large for GitHub. You may download it from:
https://nlp.stanford.edu/projects/glove/
Download `glove.6B.zip` and add `glove.6B.100d.txt` to the project folder.
