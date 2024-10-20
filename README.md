# Conversational AI 

 
Conversational ChatBot's built using Seq2Seq model, Decision Tree's & Random Forests models. Python 3.6 & tensorflow 1.0.0. Comparision of chatbot's on performance and accuracy.

# How to run:
	
To run the neural net and bayes classifier chat bot please input program name and desired bot type: “interactive_chat.py --neuralnet” or “interactive_chat.py --naivebayes” 

# To run the RNN model:

There is a pre-trained model called chatbot_weight.ckpt. If using this pre-trained model, one can run the preprocessing data, model building parts of the jupyter notebook. Then one can skip the training part and go to the testing part of the notebook. Warning: training the model will take a considerably long time if not using a GPU booster. To quit the chatbot enter the word "Goodbye" and the chatbot will exit. Trained model will not provide accurate results, trained on 3 epochs, will need further training if more accurate results are desired. 


# Multinomial Bayes Classifier

To implement a Naive Bayes based chat bot, we used a multinomial classifier, allowing each word in the training examples counted in order to be used to calculate the probability of each possible response given the evidence of some inputted string of words. Training data can be found in chatbot_data.json. For each sentence given to the bot the algorithm processes the data into a string of words, then there is a calculation for the probability of that word appearing given each certain response class (# times appeared in that specific class / total # of times appeared) at which point the probability for each was multiplied together then finally multiplied by the probability for a response of that class (# of words in that class / total # of words). This algorithm was able to run fairly quickly for each inputted response and worked fairly well for relevant questions. 

Issues faced when testing this method were errors due to overlap of less relevant words. The first fix for this was to introduce a list of “Stop words” that would prevent irrelevant contractions and such (like ‘and’, ‘is’, ‘to’, etc) to keep inaccurate responses from popping up. Ultimately though, most words must be retained and in testing it was obvious where some of these errors broke the bot. For instance asking the bot “who are you” would result in the wrong response because the probability for a complaint response won because the prior probability for complaints was higher than the correct answer, and the word “are” appeared very regularly in that category. However, asking “who is this” provoked the correct response because it did not contain the troublesome “are” issue. These issues are seen to be fixed in more complex methods for the bot. Overall this bot predicted 8 out of 15 correct testing examples. 



# Random Forests 

Random forests are constructed from decision tree algorithms. The algorithm uses ensemble learning, which combines many classifiers to produce solutions to complex problems. The algorithm consists of many decision tree's which predicts tge acerage output from the various tree's. The cahtbot follows the idea of superivsed learning classification, by gaining information from the user through intents and keywords, creating a intent classification and outputting the result. The data set used is small, therefore the random forest will not perform well outside the scope of the dataset. However, it did perform well for the data set provided. The data is preprocessed to seperate feautures, labels and intents. Scikit learn is used to transform the data to a matrix of tokens and perform the model estimation using the RandomForestClassifier module. 

![image](https://user-images.githubusercontent.com/35156624/130139945-70c499f7-f6b0-4204-ac8e-5a235515281c.png)
Figure 1.1, https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/

# Neural Net with Keras (Using limited scope training set)

For this algorithm, we used a neural net architecture from tensorflow called keras. The inputs consisted of labed examples of patterns that appear for each separate tag in the data. Each tag then had associated answers that the bot should reply with. The model used three hidden nodes, two using a relu activation function and one using a softmax. The decision for the function and the weights is somewhat arbitrary as this is a very simple bot, but these functions are some of the most common for composing neural nets and worked well for our bot. 

Overall this bot performed much better than the non-deep learning approaches. However, the largest limitation for this bot is a very small data sample size. While this implementation shows the power of an artificial neural net in classification predicting problems, it also shows the limitation of artificial intelligence to make good predictions without voluminous amounts of test data. While this bot is well trained at responding to specific requests related to its training, it still doesn’t sound quite human and its mistakes make it difficult to believe there may be a person behind it instead of a computer. Overall, the amount of data given for natural language processing would make a much more convincing bot if it had trained on not just knowledge of correct responses for this situation, but rather learned more about the English language and speaking as a human would. We explore this more with our next algorithm as we progress into training a neural net with Seq2Seq. 


# Neural Net with Seq2Seq (Using large scope training set)

![image](https://user-images.githubusercontent.com/35156624/130003456-690ecc20-7166-4da5-afcd-05ce928ccba9.png)

Figure 1.2, https://towardsdatascience.com/day-1-2-attention-seq2seq-models-65df3f49e263

For this model we used recurrent neural networks with seq2seq architecture. Tensorflow 1.0.0 and python 3.5. The model’s architecture is composed of an encoder and decoder. The encoder gathers information from the input sequence and feeds it into the decoder, which produces the output sequence. The data set used was the Cornell movie lines data set. The data is preprocessed to allow for the neural network to read the data. Since the data is categorical, the text is transformed to integer values which were gained from mapping the lines with the conversation numbers in a hash table. Lines that reached a threshold of 20 had to be removed, as this would increase the time taken for the RNN to train the data. A cross validation approach is used with a 20-80 split for training and test data. 

We set the hyperparameters to account for the time constraint of the project. We faced a drawback when training the data, as the time taken to train with 100 epochs was too large. We reduced the number of epochs to 3. There was an option to use a GPU boost through AWS or another cloud service but for the scope of this project, we opted to train on a smaller amount of epochs. As the training was only on a significantly large amount of data but only a small number of epochs the accuracy of the Seq2Seq model is not very high. This will dramatically improve as the model has more time to train. The weights are saved in a file called chatbot_weights. This file path has to have the same name in order to use the weights trained from the model. 

In order for the model to function tensorflow 1.0.0 is needed which requires python 3.6 and below. This is a fully functional conversational chatbot. It uses a many to many Seq2Seq model used by leading organizations. 

# Conclusion 

Through the progression of these bots it is obvious that deep learning techniques can greatly enhance the accuracy of predictions for chatbot classifications. However the biggest advancement in the bots comes from the increase in the size and complexity of data sets. At a certain point there are limitations of the performance of any model where the accuracy depends on the volume of training data provided. The downside however to large data sets is cost of computation for both time and cpu. This led to the biggest error in accuracy of the Seq2Seq chatbot. 
Currently, chatbots are now an essential tool for businesses to handle customers on a daily basis and there are hundreds of developers competing to produce the most efficient and intelligent bot. The field is rapidly evolving and finding new means of raising the bar: “As the market matures, 40% of chatbot/virtual assistant applications launched in 2018 will have been abandoned by 2020” (https://www.artificial-solutions.com/chatbots#14) Natural language processing is essential to creating a life-like agent, this will allow bots not just to answer questions relating to a specific cause, but also have complex conversations that capture nuances of language and human relationships. 

# Interface Examples of Chatbots

![Screen Shot 2021-08-18 at 5 47 05 PM](https://user-images.githubusercontent.com/35156624/130003646-0ed3eeb5-2505-4728-bb6a-f188c79e0ac0.png)

Figure 1.3, interface of chatbot
