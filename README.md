# Script2Screen


## Inspiration 

We were initially having trouble on which aspect of pop culture to focus our project on. Naturally, the conversation drifted and we started listing out the recent movies that we’ve seen and our thoughts on them. We all seemed to dislike the same movies, and upon research into their financials, it seems like these movies flopped for a reason. In discussing our critiques, they mainly lied within the story. Thus, our idea stemmed here. We wanted to build some studio-facing system that analyzes the success of a film based on its core screenplay. 

At the end of the day, we believe the success of a film lies in its story, and we explored this through our project.

## What it does 

Our final idea ended up being two models used at different phases of the film-making process. Firstly, we have the Ideation Model – a model meant for screenwriters and producers to simply test the potential of a synopsis, while playing around with budget and release year. The second model, the Pre-Production Model, is meant to be used right before production starts. The studio can additionally input the finalized budget, genre, production studio, and more features, to get a revenue-and-ROI-based success classification–Flop, Average, Hit, and Blockbuster!

## How we built it 

We first found a dataset from The Movies Database (TMDB) with data till 2024 and multiple revenue and production features. We cleaned it thoroughly to take out missing values in our chosen features for both models, and created new columns for profit and ROI, in order to properly train for success classification in the Pre-Production Model.

We built the frontend of our public-facing website using HTML, CSS, and JavaScript, focusing on creating a user-friendly and visually engaging experience. On the backend, we used Flask to deploy both of our machine learning models, ensuring smooth and efficient interactions for our users. For the model, we used the scikit-learn package in Python and incorporated a TFIDF Vectorizer, Binarizer, OneHotEncoder, StandardScaler, MultiLabelBinarizer, and made custom transformers.

## Challenges we ran into 

One of the biggest challenges we faced was dealing with missing or messy financial information like key details like budgets, revenues, and release dates. The large variety of movie genres and global audiences added complexity; however, it also causes the problem of potentially working in one market, but falling flat in another. From a technical standpoint, it was challenging to combine structured numerical data (like runtime and budget) with unstructured text data (like the plot synopsis). It required us to look into feature engineering to ensure that both types of information were captured meaningfully. Additionally, the limited number of movies within the larger dataset, that have the complete plot and revenue data made it harder to train models that generalize well. 

## Accomplishments that we're proud of 

Some accomplishments that we are proud of is successfully coming up with our idea, obtaining datasets, and cleaning them all in quick time. Our brainstorming process was efficient as we collaborated and combined our ideas to come up with a good topic. Next, we all worked to obtain the dataset, and discussed ways in which we could clean it to best model our idea. We believe that this method of collaboration in the initial stages could prove more efficient than splitting up the work.

## What we learned

We learned how to use Flask to build and deploy the backend/frontend of our web application, which allowed us to connect the user inputs with our machine learning models. We also gained hands-on experience with Natural Language Processing, using TfidfVectorizer to transform movie synopses into meaningful numerical representations that could be used for predictions and polarity scores to analyze the sentiment and emotion of the synopsis. By extrapolating different features from the synopsis text, we were able to get a better understanding of textual analysis overall. 

## What's next for Script2Screen

We plan to make Script2Screen even more interactive by incorporating genre-specific visualizations, confidence intervals for revenue predictions, and personalized recommendations for improving a synopsis. We are also looking for ways to integrate public datasets like box office trends or critic sentiment to further contextualize predictions. Ultimately, the goal of our project is to have Script2Screen become a go-to tool for indie creators, scriptwriters, and even studios to test and pitch ideas instantly with data-driven insights.

