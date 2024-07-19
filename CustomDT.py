import numpy as np
import pandas as pd
import math

# Read the dataset
df = pd.read_csv("E:\Πτυχιακή\datasets\dataset.csv")
col = df.columns

# Preprocessing
df = df[df['Hours'] != 0]
df['Hours'] = np.where(df['Hours'] < 3, 0, 1)
df['Kids'] = np.where(df['Kids'] != 0, 1, df['Kids'])

def unique_values(rows, col):
    """Find the unique values for a column in a dataset."""
    return set(rows[col])

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows['Hours']:  # 'hours' is the target column
        label = row
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_column_numeric(dataframe, col):
    """Check if a column in the dataframe is numeric."""
    return np.issubdtype(dataframe[col].dtype, np.number)

class Question:
    """A Question is used to partition a dataset."""
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_column_numeric(df, self.column):
            return val <= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "==" if not is_column_numeric(df, self.column) else "<="
        return f"Is {self.column} {condition} {str(self.value)}?"

def partitions(rows, question):
    """Partitions a dataset."""
    true_rows, false_rows = [], []
    for index, row in rows.iterrows():
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return pd.DataFrame(true_rows), pd.DataFrame(false_rows)

def entropy(class_counts):
    """Calculate the entropy of the given class counts."""
    total_samples = sum(class_counts.values())
    entropy_val = 0
    for count in class_counts.values():
        if count != 0:
            probability = count / total_samples
            entropy_val -= probability * math.log2(probability)
    return entropy_val

def information_gain(true_rows, false_rows, current_uncertainty):
    """Calculate the information gain."""
    p = float(len(true_rows)) / (len(true_rows) + len(false_rows))
    return current_uncertainty - p * entropy(class_counts(true_rows)) - (1 - p) * entropy(class_counts(false_rows))

def find_best_split_entropy(rows):
    """Find the best split using entropy."""
    best_gain = 0
    best_question = None
    current_uncertainty = entropy(class_counts(rows))
    for column in rows.columns:
        if column == 'Hours' or column == 'Response ID':
            continue
        for value in unique_values(rows, column):
            question = Question(column, value)
            true_rows, false_rows = partitions(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = information_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

def gini_impurity(class_counts):
    """Calculate the Gini impurity."""
    total_samples = sum(class_counts.values())
    impurity = 1
    for count in class_counts.values():
        probability = count / total_samples
        impurity -= probability ** 2
    return impurity

def gini_gain(true_rows, false_rows, current_impurity):
    """Calculate the Gini gain."""
    p = float(len(true_rows)) / (len(true_rows) + len(false_rows))
    return current_impurity - p * gini_impurity(class_counts(true_rows)) - (1 - p) * gini_impurity(class_counts(false_rows))

def find_best_split_gini(rows):
    """Find the best split using Gini impurity."""
    best_gain = 0
    best_question = None
    current_impurity = gini_impurity(class_counts(rows))
    for column in rows.columns:
        if column == 'Hours' or column == 'Response ID':
            continue
        for value in unique_values(rows, column):
            question = Question(column, value)
            true_rows, false_rows = partitions(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = gini_gain(true_rows, false_rows, current_impurity)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

class Leaf:
    """A Leaf node classifies data."""
    def __init__(self, rows):
        self.predictions = class_counts(rows)
    
    def print_leaf(self):
        total = sum(self.predictions.values())
        probabilities = {k: v / total for k, v in self.predictions.items()}
        return probabilities

class Decision_Node:
    """A Decision Node asks a question."""
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows, impurity_criterion='entropy', min_samples_split=2, max_depth=float('inf'), curr_depth=0):
    """Build the decision tree."""
    if impurity_criterion == 'entropy':
        gain, question = find_best_split_entropy(rows)
    else:
        gain, question = find_best_split_gini(rows)

    if gain == 0 or len(rows) < min_samples_split or curr_depth >= max_depth:
        return Leaf(rows)

    true_rows, false_rows = partitions(rows, question)
    true_branch = build_tree(true_rows, impurity_criterion, min_samples_split, max_depth, curr_depth + 1)
    false_branch = build_tree(false_rows, impurity_criterion, min_samples_split, max_depth, curr_depth + 1)

    return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):
    """Print the tree."""
    if isinstance(node, Leaf):
        print(spacing + f"Predict {node.print_leaf()}")
        return

    print(spacing + f"{node.question}")
    print(spacing + f"|--- True:")
    print_tree(node.true_branch, spacing + "    ")
    print(spacing + f"|--- False:")
    print_tree(node.false_branch, spacing + "    ")

def classify(row, node):
    """Classify a given row with the tree."""
    if isinstance(node, Leaf):
        return node.print_leaf()
    
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def train_test_split(df, test_size, random_state):
    np.random.seed(random_state)
    indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    
    return train_df, test_df

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=4)

# Build the tree
decision_tree = build_tree(train_df, impurity_criterion='entropy', min_samples_split=10, max_depth=5)

# Print the tree
# print_tree(decision_tree)


predictions = {}


y_true = test_df['Hours']
y_pred = [max(classify(row, decision_tree), key=classify(row, decision_tree).get) for index, row in test_df.iterrows()]

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0

def recall_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0

def f1_score(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0

for index, row in test_df.iterrows():
    response_id = row['Response ID']
    prediction = classify(row, decision_tree)
    predicted_class = max(prediction, key=prediction.get)
    actual_class = int(row['Hours'])
    predictions[response_id] = (predicted_class, actual_class)
    
    
# Function for 10-fold cross-validation
def cross_validation(df, impurity_criterion='entropy', min_samples_split=10, max_depth=5, k=10):
    np.random.seed(4)
    indices = np.random.permutation(len(df))
    fold_size = len(df) // k
    
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for fold in range(k):
        test_indices = indices[fold * fold_size : (fold + 1) * fold_size]
        train_indices = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])

        train_df = df.iloc[train_indices]
        test_df = df.iloc[test_indices]

        # Build the tree
        decision_tree = build_tree(train_df, impurity_criterion, min_samples_split, max_depth)

        # Make predictions on the test set
        predictions = {}
        for index, row in test_df.iterrows():
            response_id = row['Response ID']
            prediction = classify(row, decision_tree)
            predicted_class = max(prediction, key=prediction.get)
            actual_class = int(row['Hours'])
            predictions[response_id] = (predicted_class, actual_class)

        # Calculate evaluation metrics
        y_true = np.array([actual_class for _, (_, actual_class) in predictions.items()])
        y_pred = np.array([predicted_class for _, (predicted_class, _) in predictions.items()])

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Append scores for this fold
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # Print scores for this fold
        print(f"Fold {fold + 1}:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print()

    # Compute average scores
    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    return avg_accuracy, avg_precision, avg_recall, avg_f1

# Perform 10-fold cross-validation
avg_accuracy, avg_precision, avg_recall, avg_f1 = cross_validation(df, impurity_criterion='entropy', min_samples_split=10, max_depth=5, k=10)

# Print average evaluation metrics
print("Average Evaluation Metrics:")
print(f"Average Accuracy: {avg_accuracy}")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average F1 Score: {avg_f1}")
