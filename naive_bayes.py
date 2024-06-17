# Nicholas Moreland
# 1001886051

import numpy as np

# Load data from a file and return it as an array
def loadFile(file_path):
    data = np.loadtxt(file_path)
    return data

# Calculate the mean and standard deviation for each feature
def calculateStats(data):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0, ddof=1)
    
    # Making sure the std is not smaller than 0.01
    stds[stds < 0.01] = 0.01
    
    return means, stds

# Calculate the mean and standard deviation for each class and feature
def calculateClassStats(data):
    # Determine the number of classes and features
    num_classes = len(np.unique(data[:, -1]))
    
    # Exclude the last column (class labels)
    num_features = data.shape[1] - 1
    
    # Initialize empty arrays to store class-specific means and standard deviations
    class_means = np.zeros((num_classes, num_features))
    class_stds = np.zeros((num_classes, num_features))
    
    # Calculate means and standard deviations for each class
    for class_label in range(1, num_classes + 1):
        class_data = data[data[:, -1] == class_label][:, :-1]
        class_means[class_label - 1], class_stds[class_label - 1] = calculateStats(class_data)
    
    return class_means, class_stds

# Calculate the probability of a value given a mean and standard deviation
def gaussianProbability(x, mean, std):
    exponent = -((x - mean) ** 2) / (2 * (std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(exponent)

# Normalize the probabilities to ensure they sum to 1
def sumProbabilities(probabilities):
    total = np.sum(probabilities)
    return probabilities / total

# Naive Bayes classifier
def naive_bayes(training_file, test_file):
    # Load training and test data
    training_data = loadFile(training_file)
    test_data = loadFile(test_file)
    
    # Calculate mean and standard deviation for each class and feature separately
    class_means, class_stds = calculateClassStats(training_data)
    
    # Print mean and standard deviation for each dimension and class
    num_classes = len(class_means)
    num_features = class_means.shape[1]
    for class_label in range(1, num_classes + 1):
        for attribute in range(num_features):
            print(f"Class {class_label}, attribute {attribute + 1}, mean = {class_means[class_label - 1, attribute]:.2f}, std = {class_stds[class_label - 1, attribute]:.2f}")
    
    # Initialize variables to store results
    num_correct = 0
    num_total = len(test_data)
    
    # Classify test objects and get classifications
    classifications = []
    
    for i, test_instance in enumerate(test_data):
        instance_features = test_instance[:-1]
        probabilities = []
        
        # Calculate the probability for each class
        for class_label in range(1, num_classes + 1):
            class_mean = class_means[class_label - 1]
            class_std = class_stds[class_label - 1]
            class_probability = np.prod(gaussianProbability(instance_features, class_mean, class_std))
            
            # Multiply by the class prior probability
            class_probability *= (np.sum(training_data[:, -1] == class_label) / len(training_data))
            
            probabilities.append(class_probability)
        
        # Normalize the probabilities to ensure they sum to 1
        normalized_probabilities = sumProbabilities(probabilities)
        
        # Predict the class with the highest probability (startiing from 1)
        predicted_class = np.argmax(normalized_probabilities) + 1
        
        # If predicted class is 0, set it to 1
        if predicted_class == 0:
            predicted_class = 1
        
        # Calculate accuracy
        correct_class = int(test_instance[-1])
        accuracy = 1 if predicted_class == correct_class else 0
        
        # Calculate overall classification accuracy
        classifications.append((i + 1, predicted_class, normalized_probabilities[predicted_class - 1], correct_class, accuracy))
    
    # Print the results for each test object
    for i, predicted_class, probability, correct_class, accuracy in classifications:
        print(f"ID={i:5d}, predicted={predicted_class:3d}, probability = {probability:.4f}, true={correct_class:3d}, accuracy={accuracy:.2f}")
        
        # Check if the prediction is correct
        if predicted_class == correct_class:
            num_correct += 1
    
    # Calculate and print the overall classification accuracy
    overall_accuracy = num_correct / num_total
    print(f"classification accuracy={overall_accuracy:.4f}")







