# RedBlue

RedBlue is a political language classifier for news articles. It uses Baleen
(https://github.com/bbengfort/baleen) to ingest RSS feeds into MongoDB. We
then parse the data, remove stop words, and vectorize the data. Once
it's in the proper format, we pass it to a scikit-learn logistic regression
algorithm that we trained using transcripts from the 2016 presidential
primary debates.
