# Finding out the political affiliation of users on Reddit

Two datasets of comments were collected from Reddit in March 2020 – one from the subreddit /r/JoeBiden of supporters of Joe Biden and one from the subreddit /r/The_Donald of supporters of Donald Trump. Here, our goal is to learn if those two groups speak the same type of language.

For this task, two embeddings of the comments were created: one with a pre-trained BERT model, and the other with a Doc2Vec model trained over the comments. Then, a randomforest was trained over each embedding in order to learn to separate the comments of one subreddit from another, i.e. to identify the political affiliation of users'. It was found that the model trained over the BERT embedding achieved better performances even though the BERT model never saw the comments before, whereas the Doc2Vec model did.

Train accuracy: 0.89 for the Doc2Vec embedding, 0.937 for the BERT embedding
Test accuracy: 0.58 for the Doc2Vec embedding, 0.747 for the BERT embedding
