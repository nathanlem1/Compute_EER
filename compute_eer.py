"""
This code computes False Acceptance Rate (FAR), False Rejection Rate (FRR) and Equal Error Rate (EER) given similarity
scores of a biometric system and the actual label of the sample (genuine or imposter). It also plots some visualizations
such as histogram of genuine (blue) and imposter (red) distributions with bin width of 0.05, EER along with FAR and FRR,
and ROC curve.
The following terms are crucial for understanding:
True acceptance - true positive
False acceptance - false positive
True rejection - true negative
False rejection - false negative

The following link is very useful for understanding how biometric (face recognition) system is developed and evaluated.
https://becominghuman.ai/face-recognition-system-and-calculating-frr-far-and-eer-for-biometric-system-evaluation-code-2ac2bd4fd2e5
https://alicebiometrics.com/en/defining-the-core-accuracy-metrics-of-biometric-systems/


Author: Nathanael L. Baisa
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def compute_far(imposter_scores):
    """
    This computes False Acceptance Rate (FAR).

    Input:
        imposter_scores: similarity score for imposter (forgery).
    Output:
        far: list that we will save the FAR in each threshold.
        threshold: list of threshold and it will go from 0% to 100% i.e. from 0 to 1 (normalized) for our plot.
    """
    far = []
    threshold = []
    for i in range(100):
        # Count how many imposters (forgeries) will pass at each threshold.
        count = 0
        for x in imposter_scores:
            if x > i / 100:
                count += 1
        far.append(count)
        threshold.append(i)

    far = np.array(far) / 100  # Normalize
    threshold = np.array(threshold) / 100  # Normalize

    return far, threshold


def compute_frr(genuine_scores):
    """
    This computes False Rejection Rate (FRR).

    Input:
        genuine_scores: similarity score for genuine.
    Output:
        frr: a list that we will save the FRR in each threshold.
        threshold: list of threshold and it will go from 0% to 100% i.e. from 0 to 1 (normalized) for our plot.
    """
    frr = []
    threshold = []
    for i in range(100):
        # Count how many genuines get rejected at each threshold.
        count = 0
        for x in genuine_scores:
            if x < i / 100:
                count += 1
        frr.append(count)
        threshold.append(i)

    frr = np.array(frr) / 100  # Normalize
    threshold = np.array(threshold) / 100  # Normalize

    return frr, threshold


def compute_eer(far, frr):
    """
    This computes Equal Error Rate (EER). EER is the point where the FAR and FRR meet or closest to each other, and it
    represents the best threshold to choose. The smaller is the better.

    Input:
        far: false accept rate.
        frr: false rejection rate.
    Output:
        best_threshold: best threshold to choose which corresponds to EER.
        eer: EER at the best threshold.
    """
    diffs = np.abs(far - frr)
    min_index = np.argmin(diffs)
    eer = np.mean((far[min_index], frr[min_index]))  # eer = (far + frr)/2 at min_index.
    best_threshold = min_index / 100

    return best_threshold, eer


# Main function
def main():
    # Read the data
    df = pd.read_excel("similarity_scores.xlsx")  # df has 'similarity score' & 'genuine (1)/imposter (0)' columns.
    print(df)
    data_np = df.to_numpy()
    similarity_scores = data_np[:, 0]
    actual_labels = data_np[:, 1]
    genuine_labels_scores = data_np[data_np[:, 1] == 1]
    imposter_labels_scores = data_np[data_np[:, 1] == 0]
    genuine_scores = genuine_labels_scores[:, 0]
    imposter_scores = imposter_labels_scores[:, 0]

    # Plot histogram of genuine (blue) and imposter (red) distributions, with bin width of 0.05.
    bin_width = 0.05
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_scores, edgecolor='blue', bins=np.arange(min(genuine_scores), max(genuine_scores) + bin_width,
                                                              bin_width))
    plt.hist(imposter_scores, edgecolor='red', bins=np.arange(min(imposter_scores), max(imposter_scores) + bin_width,
                                                              bin_width))
    plt.legend(['genuine', 'imposter'], loc='upper center', shadow=True, fontsize='large')
    plt.title('Histogram of genuine (blue) and imposter (red) distributions, with bin width of 0.05.')
    plt.show()

    # Compute FAR
    far, threshold_far = compute_far(imposter_scores)
    print('FAR: ', far)

    # Compute FRR
    frr, threshold_frr = compute_frr(genuine_scores)
    print('FRR: ', frr)

    # Compute EER
    best_threshold, eer = compute_eer(far, frr)
    print('Best threshold and EER = ', (best_threshold, eer))

    # Plot the FRR, FAR & EER.
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_far, far, 'r-', label='FAR')
    plt.plot(threshold_frr, frr, 'b-', label='FRR')
    plt.xlabel('Threshold')
    plt.plot(best_threshold, float(eer), 'go', markersize=10, label='EER')
    plt.legend(['FAR', 'FRR', 'EER'], loc='upper center', shadow=True, fontsize='large')
    plt.show()

    # Plot ROC Curve using only FAR and FRR
    plt.figure(figsize=(10, 6))
    tar = 1 - frr  # True acceptance rate
    plt.plot(far, tar, 'b-')
    plt.xlabel('False positive rate (false acceptance rate)')
    plt.ylabel('True positive rate (true acceptance rate)')
    plt.title('ROC Curve')
    plt.show()

    # Plot ROC Curve using similarity scores and actual labels along with sklearn.metrics
    fpr, tpr, thresholds = roc_curve(actual_labels, similarity_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, 'b-')
    plt.xlabel('False positive rate (false acceptance rate)')
    plt.ylabel('True positive rate (true acceptance rate)')
    plt.title('ROC Curve')
    plt.show()


# Execute main function
if __name__ == "__main__":
    main()
