import numpy as np

def pca(data, n):
    # Schritt 1: Daten zentrieren
    X_meaned = data - np.mean(data, axis=0)

    # Schritt 2: Daten standardisieren
    X_std = np.std(X_meaned, axis=0)
    X_standardized = X_meaned / X_std

    # Schritt 3: Singulärwertzerlegung
    U, Sigma, VT = np.linalg.svd(X_standardized)

    # Schritt 4: Auswahl der ersten n Hauptkomponenten
    V = VT.T[:, :n]

    # Schritt 5: Projektion der Daten auf die Hauptkomponenten
    X_pca = np.dot(X_standardized, V)

    return X_pca, Sigma[:n], V