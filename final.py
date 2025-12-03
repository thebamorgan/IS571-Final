import nltk, random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import io
from contextlib import redirect_stdout
from nltk.corpus import stopwords
import string

# It's good practice to ensure the necessary NLTK data is downloaded.
nltk.download('movie_reviews', quiet=True)
nltk.download('stopwords', quiet=True)

# 1. How many words are there in this corpus?
total_words = len(movie_reviews.words())
print(f"Total number of words in the movie_reviews corpus: {total_words}")

# 2. What are the two movie review categories?
categories = movie_reviews.categories()
print(f"The categories are: {categories}")

# 3. For more details about this corpus, run movie_reviews.readme().
print("\n--- Corpus README ---")
# Capture the output of .readme() to print it cleanly.
with io.StringIO() as buf, redirect_stdout(buf):
    movie_reviews.readme()
    readme_output = buf.getvalue()
print(readme_output)

# 4. Create a list of documents with words and categories
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# 5. Randomly shuffle the documents
random.shuffle(documents)

print("\nSuccessfully created and shuffled the 'documents' list.")
print(f"Total number of documents: {len(documents)}")
print("Example of one document (first 15 words and its category):")
print(f"Words: {documents[0][0][:15]}")
print(f"Category: {documents[0][1]}")

# 6. Create a list of the 2000 most frequent words

# First, get a list of all words in the corpus, converted to lowercase.
all_words_raw = movie_reviews.words()

# Define English stopwords and punctuation to be excluded.
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Filter out stopwords and punctuation from our list of words.
filtered_words = [w.lower() for w in all_words_raw if w.lower() not in stop_words and w.lower() not in punctuation]

# Create a frequency distribution of the filtered words.
all_words_freq = nltk.FreqDist(filtered_words)

# Create the final list of the 2000 most common words.
word_features = list(all_words_freq.keys())[:2000]

print("\nSuccessfully created 'word_features' list with the 2000 most frequent words.")
print("Example features (first 15):")
print(word_features[:15])



# 7. Define the feature extractor function
def document_features(document):
    """
    Checks whether each of the 2000 most frequent words is contained in a document.
    Returns a dictionary of boolean features.
    """
    document_words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)
    return features

# 8. Create the feature sets by applying the function to each document
featuresets = [(document_features(d), c) for (d, c) in documents]

print("\nSuccessfully created 'featuresets' list.")
print(f"Total number of feature sets: {len(featuresets)}")
print("\nExample of one feature set (first 5 features and category):")
print(f"Features: {list(featuresets[0][0].items())[:5]}")
print(f"Category: {featuresets[0][1]}")

# 9. Split the feature sets into training and testing sets
# The test set will be the first 100 documents, and the training set will be the remaining 1900.
test_set = featuresets[:100]
train_set = featuresets[100:]

print("\nSplitting data: 1900 for training, 100 for testing.")

# 10. Train the Naive Bayes classifier on the training set
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Naive Bayes classifier trained successfully.")

# 11. Evaluate the classifier's accuracy on the test set
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"\nOut-of-sample prediction accuracy for the test set: {accuracy:.4f}")

# 12. Display the top 15 most informative features
print("\nTop 15 most informative features for the classifier:")
classifier.show_most_informative_features(15)

# 13. Perform twenty-fold cross-validation to get a more robust accuracy measure.

# A list to store the accuracy rate of each fold.
accuracy_rates = []
num_folds = 20
# Each fold will contain 100 feature sets (2000 total / 20 folds).
subset_size = len(featuresets) // num_folds

print("\n--- Twenty-Fold Cross-Validation ---")

for i in range(num_folds):
    # Define the start and end indices for the current test set fold.
    test_start = i * subset_size
    test_end = (i + 1) * subset_size
    
    # The test set is the current fold; the training set is everything else.
    test_set_fold = featuresets[test_start:test_end]
    train_set_fold = featuresets[:test_start] + featuresets[test_end:]
    
    # Train a new classifier on the training set for this fold.
    classifier_fold = nltk.NaiveBayesClassifier.train(train_set_fold)
    # Calculate the accuracy for this fold and add it to our list.
    accuracy_fold = nltk.classify.accuracy(classifier_fold, test_set_fold)
    accuracy_rates.append(accuracy_fold)
    print(f"Fold {i+1:2d} out-of-sample prediction accuracy: {accuracy_fold:.4f}")

# Calculate and display the average accuracy across all 20 folds.
average_accuracy = sum(accuracy_rates) / len(accuracy_rates)
print(f"\nOverall prediction accuracy (average of 20 folds): {average_accuracy:.4f}")
