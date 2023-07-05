import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score,  roc_auc_score
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report



def task_6(tr):
    dot_data = export_graphviz(tr, out_file=None, feature_names=X_train.columns, filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render(filename='tree', format='png')
    graph.view()


def task_8(name):
    max_leaf_nodes_range = range(7, 51, 3)
    min_samples_leaf_range = range(60, 1760, 125)

    scores = np.zeros((len(max_leaf_nodes_range), len(min_samples_leaf_range)))

    for i, max_leaf_nodes in enumerate(max_leaf_nodes_range):
        for j, min_samples_leaf in enumerate(min_samples_leaf_range):
            tr = tree.DecisionTreeClassifier(criterion=name, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)
            scores[i, j] = np.mean(cross_val_score(tr, X, y, cv=10))

    fig, ax = plt.subplots()
    im = ax.imshow(scores, cmap='cool')

    # Set ticks and labels for x and y axis
    ax.set_xticks(np.arange(len(min_samples_leaf_range)))
    ax.set_yticks(np.arange(len(max_leaf_nodes_range)))
    ax.set_xticklabels(min_samples_leaf_range)
    ax.set_yticklabels(max_leaf_nodes_range)
    ax.set_xlabel('Min samples leaf')
    ax.set_ylabel('Max leaf nodes')
    ax.set_title('Classification scores')


    for i in range(len(max_leaf_nodes_range)):
        for j in range(len(min_samples_leaf_range)):
            text = ax.text(j, i, "{:.2f}".format(scores[i, j]), ha="center", va="center", color="black")


    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Mean accuracy', rotation=-90, va="bottom")


    plt.show()


def task_9(tr, X_train):
    best_atr = tr.feature_importances_
    best_atr_df = pd.DataFrame({'features': list(X_train), 'feature_importance': best_atr})
    best_atr_df = best_atr_df.sort_values('feature_importance', ascending=False)

    sns.set_color_codes("muted")
    ax = sns.barplot(x="feature_importance", y="features", data=best_atr_df, label="importance", color="b")
    ax.set_title("The importance of attributes")
    ax.set_xlabel("feature importance")
    ax.set_ylabel("features")
    plt.show()


def metric_calculation(y_pred_train, y_pred):
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"Train accuracy: {train_acc:.6f}")
    print(f"Test accuracy: {test_acc:.6f}")

    train_precision = precision_score(y_train, y_pred_train, average='micro')
    test_precision = precision_score(y_test, y_pred, average='micro')

    print(f"Train precision: {train_precision:.6f}")
    print(f"Test precision: {test_precision:.6f}")

    train_recall = recall_score(y_train, y_pred_train, average='micro')
    test_recall = recall_score(y_test, y_pred, average='micro')

    print(f"Train recall: {train_recall:.6f}")
    print(f"Test recall: {test_recall:.6f}")

    train_f1 = f1_score(y_train, y_pred_train, average='micro')
    test_f1 = f1_score(y_test, y_pred, average='micro')

    print(f"Train F1-score: {train_f1:.6f}")
    print(f"Test F1-score: {test_f1:.6f}")

    mcc_train = matthews_corrcoef(y_train, y_pred_train)
    mcc_test = matthews_corrcoef(y_test, y_pred)

    print(f"Train Matthews Correlation Coefficient (MCC): {mcc_train:.6f}")
    print(f"Test Matthews Correlation Coefficient (MCC): {mcc_test:.6f}")

    ba_train = balanced_accuracy_score(y_train, y_pred_train)
    ba_test = balanced_accuracy_score(y_test, y_pred)

    print(f"Balanced Accuracy (BA) : {ba_train:.6f}")
    print(f"Balanced Accuracy (BA) : {ba_test:.6f}")

    auc_train = roc_auc_score(y_train, tr_entropy.predict_proba(X_train)[:, 1])
    auc_test = roc_auc_score(y_test, tr_entropy.predict_proba(X_test)[:, 1])

    print(f"Balanced Area Under Curve for Receiver Operation Curve : {auc_train:.6f}")
    print(f"Balanced Area Under Curve for Receiver Operation Curve : {auc_test:.6f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.show()
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    nameCol = ['column_' + str(i) for i in range(1, 12)]
    df = pd.read_csv("dataset3.csv", header=None, names=nameCol)
    # 1
    print(df.head())

    # 2
    print('Size data frame:', df.shape)

    # 3
    print('first 10:\n', df.head(10))

    # 4
    X = df.drop('column_11', axis=1)
    y = df['column_11']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=47)

    #5-6
    #ENTROPY
    print('-' * 35, '\nENTROPY BASED DECISION TREE')
    tr_entropy = tree.DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_leaf=200, random_state=15)
    tr_entropy.fit(X_train, y_train)
    y_pred_entropy = tr_entropy.predict(X_test)
    y_pred_train_entropy = tr_entropy.predict(X_train)

    task_6(tr_entropy)

    #task 7
    metric_calculation(y_pred_train_entropy, y_pred_entropy)

    #task_8
    task_8("entropy")

    #task_9
    task_9(tr_entropy,X_train)

    # 5-6
    #GINNY
    print('-' * 35, '\nGINNY BASED DECISION TREE')
    tr_gini = tree.DecisionTreeClassifier(criterion="gini", max_depth=6, min_samples_leaf=200, random_state=15)
    tr_gini.fit(X_train, y_train)
    y_pred_gini = tr_gini.predict(X_test)
    y_pred_train_gini = tr_gini.predict(X_train)

    task_6(tr_gini)

    # task 7
    metric_calculation(y_pred_train_gini, y_pred_gini)

    task_8("gini")

    task_9(tr_gini,X_train)
